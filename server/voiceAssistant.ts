import { ThinkingLevel, Type } from '@google/genai';
import {
  SERVICES_FORM_ENUMS,
  type OfficialServiceType,
  type VoiceServiceDraft,
} from '../src/config/forms.ts';
import { SITE_INFO } from '../src/config/site.ts';
import {
  AssistantServiceError,
  createGeminiClient,
  isModelInCooldown,
  markModelCooldown,
} from './assistant.ts';

export interface VoiceTurnInput {
  role: 'user' | 'assistant';
  content: string;
}

export interface VoiceAnalyzeResult {
  assistantMessage: string;
  updatedDraft: VoiceServiceDraft;
  missingFields: string[];
  readyForConfirmation: boolean;
}

const ALLOWED_AUDIO_MIME_TYPES = new Set([
  'audio/webm',
  'audio/mp4',
  'audio/ogg',
  'audio/wav',
  'audio/mpeg',
  'audio/mp3',
  'audio/aac',
  'audio/x-m4a',
  'audio/m4a',
  'audio/3gpp',
]);

export function normalizeAudioMimeType(rawMime: string): string | null {
  const base = rawMime.split(';')[0]?.trim().toLowerCase() ?? '';
  if (!ALLOWED_AUDIO_MIME_TYPES.has(base)) return null;
  if (base === 'audio/x-m4a' || base === 'audio/m4a') return 'audio/mp4';
  return base;
}

function pickExactEnum<T extends readonly string[]>(
  value: unknown,
  allowed: T
): T[number] | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return allowed.find((item) => item === trimmed);
}

function matchDeviceEnum(
  value: unknown,
  matchers: Array<{ value: string; keywords: string[] }>
): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const lower = trimmed.toLowerCase();
  for (const item of matchers) {
    if (trimmed === item.value) return item.value;
    if (item.keywords.some((kw) => lower.includes(kw))) {
      return item.value;
    }
  }
  return undefined;
}

function sanitizeDraft(
  raw: Partial<VoiceServiceDraft> | null | undefined
): VoiceServiceDraft {
  if (!raw || typeof raw !== 'object') return {};

  const serviceType = (pickExactEnum(
    raw.serviceType,
    SERVICES_FORM_ENUMS.serviceTypes
  ) ||
    matchDeviceEnum(raw.serviceType, [
      {
        value: 'تركيب راوتر أو جهاز',
        keywords: ['تركيب', 'راوتر', 'مودم', 'جهاز', 'لاقط'],
      },
      {
        value: 'الدعم الفني',
        keywords: ['دعم', 'مشكلة', 'بطء', 'انقطاع', 'فني'],
      },
      {
        value: 'صيانة وإصلاح الأعطال',
        keywords: ['صيانة', 'عطل', 'اصلاح', 'إصلاح', 'خربان'],
      },
      { value: 'نقل الخدمة', keywords: ['نقل'] },
    ])) as OfficialServiceType | undefined;

  const cleanStr = (val: unknown, maxLen: number): string | undefined => {
    if (typeof val !== 'string') return undefined;
    const t = val.trim();
    return t.length > 0 ? t.slice(0, maxLen) : undefined;
  };

  const routerDevice = matchDeviceEnum(raw.routerDevice, [
    {
      value: 'A1 – ONE (TP-Link)',
      keywords: ['tp-link', 'tplink', 'tp link', 'تي بي لينك', 'a1', 'one'],
    },
    {
      value: 'A2 – TWO (راوتر لاسلكي)',
      keywords: ['totolink', 'توتولينك', 'toto', 'a2', 'two'],
    },
    { value: 'A3 – THREE (مودم KT)', keywords: ['kt', 'كي تي', 'a3', 'three'] },
    { value: 'A4 – FOUR (مودم LG)', keywords: ['lg', 'ال جي', 'إل جي', 'a4', 'four'] },
  ]);

  const linkDevice = matchDeviceEnum(raw.linkDevice, [
    {
      value: 'B4 – NanoStation M5',
      keywords: ['nanostation m5', 'm5', 'b4', 'نانوستيشن 5'],
    },
    {
      value: 'B1 – NanoStation',
      keywords: ['nanostation', 'm2', 'b1', 'نانوستيشن', 'نانو ستيشن'],
    },
    { value: 'B2 – PowerBeam', keywords: ['powerbeam', 'b2', 'باوربيم', 'باور بيم'] },
    { value: 'B3 – LiteBeam', keywords: ['litebeam', 'b3', 'لايت بيم', 'لايتبيم'] },
  ]);

  let deviceCategory =
    pickExactEnum(raw.deviceCategory, SERVICES_FORM_ENUMS.deviceCategories) ||
    matchDeviceEnum(raw.deviceCategory, [
      { value: 'أجهزة الراوتر والمودم (A)', keywords: ['راوتر', 'مودم', '(a)'] },
      { value: 'أجهزة الربط واللاقِطات (B)', keywords: ['ربط', 'لاقط', 'خارجي', '(b)'] },
    ]);

  if (!deviceCategory) {
    if (routerDevice) deviceCategory = 'أجهزة الراوتر والمودم (A)';
    else if (linkDevice) deviceCategory = 'أجهزة الربط واللاقِطات (B)';
  }

  return {
    serviceType: serviceType ?? '',
    fullName: cleanStr(raw.fullName, 120),
    phone: cleanStr(raw.phone, 40),
    district: cleanStr(raw.district, 140),
    subscriberStatus:
      pickExactEnum(
        raw.subscriberStatus,
        SERVICES_FORM_ENUMS.subscriberStatuses
      ) ||
      matchDeviceEnum(raw.subscriberStatus, [
        { value: 'نعم، لدي اشتراك حالي.', keywords: ['نعم', 'حالي', 'مشترك حال'] },
        {
          value: 'لا، أريد الاشتراك لأول مرة.',
          keywords: ['لا', 'جديد', 'أول مرة', 'اول مرة'],
        },
        { value: 'كنت مشتركاً سابقاً.', keywords: ['سابق'] },
      ]),
    deviceCategory,
    routerDevice,
    linkDevice,
    supportProblemType:
      pickExactEnum(
        raw.supportProblemType,
        SERVICES_FORM_ENUMS.supportProblems
      ) ||
      matchDeviceEnum(raw.supportProblemType, [
        {
          value: 'انقطاع الإنترنت بالكامل.',
          keywords: ['انقطاع', 'مقطوع', 'ما يشتغل', 'لا يعمل', 'طافي'],
        },
        { value: 'بطء سرعة الإنترنت.', keywords: ['بطء', 'بطيء', 'ضعيف السرعة', 'ثقيل'] },
        {
          value: 'ضعف إشارة الواي فاي.',
          keywords: ['واي فاي', 'ضعف الإشارة', 'الاشارة', 'اشارة'],
        },
        {
          value: 'عدم القدرة على الاتصال بالشبكة.',
          keywords: ['عدم القدرة', 'اتصال بالشبكة', 'ما يشبك'],
        },
        { value: 'مشكلة في الراوتر أو المودم.', keywords: ['راوتر', 'مودم'] },
        { value: 'مشكلة أخرى.', keywords: ['أخرى', 'اخرى'] },
      ]),
    details: cleanStr(raw.details, 400),
    maintenanceDeviceName: cleanStr(raw.maintenanceDeviceName, 120),
    currentDistrict: cleanStr(raw.currentDistrict, 140),
    newDistrict: cleanStr(raw.newDistrict, 140),
  };
}

function extractPhoneFallback(text: string): string | undefined {
  const match = text.match(/(?:\+?967[\s-]*)?(7[01378]\d{7})\b/);
  return match?.[1];
}

function computeMissingFields(draft: VoiceServiceDraft): string[] {
  const missing: string[] = [];
  if (!draft.serviceType) missing.push('نوع الخدمة');
  if (draft.serviceType === 'نقل الخدمة') {
    if (!draft.currentDistrict && !draft.district) missing.push('المنطقة الحالية');
    if (!draft.newDistrict) missing.push('المنطقة الجديدة');
  } else if (!draft.district) {
    missing.push('المنطقة أو الحي');
  }
  if (!draft.fullName) missing.push('الاسم');
  if (!draft.phone) missing.push('رقم الهاتف');
  return missing;
}

export async function transcribeArabicVoice(params: {
  audioBase64: string;
  mimeType: string;
}): Promise<string> {
  const cleanMime = normalizeAudioMimeType(params.mimeType);
  if (!cleanMime) {
    throw new AssistantServiceError(
      `Unsupported audio mimeType: ${params.mimeType}`,
      400,
      'صيغة الصوت غير مدعومة. يمكنك المحاولة مرة أخرى أو كتابة طلبك نصيًا.'
    );
  }

  const ai = createGeminiClient(
    'عذرًا، خدمة التعرف الصوتي غير مهيأة حاليًا. يمكنك كتابة طلبك نصيًا مباشرة.'
  );

  const promptText =
    'فرّغ الكلام البشري المنطوق باللغة العربية (بما في ذلك اللهجة اليمنية والخليجية) في هذا التسجيل الصوتي بدقة وأمانة. أرجع النص المنطوق فقط بدون أي مقدمات أو شروحات أو علامات تنسيق. إذا كان التسجيل صامتًا أو عبارة عن ضوضاء فقط ولا يحتوي على كلام بشري واضح، أرجع نصًا فارغًا تمامًا ولا تخمّن أي عبارة.';

  const candidateModels = [
    'gemini-3.5-transcribe',
    'gemini-4-hydrogen',
    'gemini-3.1-flash-lite',
  ] as const;

  const available = candidateModels.filter((m) => !isModelInCooldown(m));
  const modelsToTry =
    available.length > 0 ? [...available] : [...candidateModels];

  let lastError: unknown;
  let lastStatus = 502;

  for (const modelName of modelsToTry) {
    try {
      const response = await ai.models.generateContent({
        model: modelName,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: cleanMime,
                data: params.audioBase64,
              },
            },
            { text: promptText },
          ],
        },
        config: {
          abortSignal: AbortSignal.timeout(18_000),
        },
      });

      const text = (response.text ?? '').trim();
      const isTimestampOrPunctuationOnly =
        text.length === 0 ||
        /^[\d:.\s\-–—_[\]()]+$/.test(text) ||
        text === '""' ||
        text === "''";
      if (isTimestampOrPunctuationOnly) {
        throw new AssistantServiceError(
          `Empty or non-speech transcription result: "${text}"`,
          422,
          'لم نلتقط كلامًا واضحًا في التسجيل. يرجى التحدث بوضوح وإعادة المحاولة، أو كتابة طلبك مباشرة.'
        );
      }
      return text;
    } catch (error: unknown) {
      if (error instanceof AssistantServiceError) throw error;
      lastError = error;
      const errObj = error as { status?: number; message?: string };
      const status = errObj?.status;
      const errMsg = String(errObj?.message ?? error ?? '');

      if (status === 429 || errMsg.includes('RESOURCE_EXHAUSTED')) {
        lastStatus = 429;
        markModelCooldown(modelName, 30_000);
        console.warn(`[VoiceTranscribe] ${modelName} quota reached (429); trying fallback.`);
        continue;
      }

      if (status === 404 || errMsg.includes('NOT_FOUND')) {
        markModelCooldown(modelName, 15 * 60_000);
        console.warn(`[VoiceTranscribe] ${modelName} unavailable (404); trying fallback.`);
        continue;
      }

      console.error(`[VoiceTranscribe] ${modelName} error (${status ?? 'unknown'}):`, errMsg);
    }
  }

  const msg = lastError instanceof Error ? lastError.message : String(lastError);
  if (lastStatus === 429) {
    throw new AssistantServiceError(
      `Transcription quota exceeded: ${msg}`,
      429,
      'تشهد خدمة تفريغ الصوت ضغطًا مؤقتًا. يمكنك المحاولة بعد قليل أو كتابة طلبك نصيًا بالأسفل.'
    );
  }
  throw new AssistantServiceError(
    `Transcription error: ${msg}`,
    502,
    'تعذر تحويل الصوت إلى نص حاليًا. يمكنك إعادة المحاولة أو كتابة طلبك نصيًا.'
  );
}

function parseVoiceJson(rawText: string): Partial<VoiceServiceDraft> & {
  assistantMessage?: unknown;
  readyForConfirmation?: unknown;
} {
  let clean = rawText.trim();
  if (clean.startsWith('```')) {
    clean = clean
      .replace(/^```(?:json)?\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim();
  }
  try {
    return JSON.parse(clean) as Partial<VoiceServiceDraft> & {
      assistantMessage?: unknown;
      readyForConfirmation?: unknown;
    };
  } catch {
    const startIdx = clean.indexOf('{');
    const endIdx = clean.lastIndexOf('}');
    if (startIdx >= 0 && endIdx > startIdx) {
      return JSON.parse(clean.slice(startIdx, endIdx + 1)) as Partial<VoiceServiceDraft> & {
        assistantMessage?: unknown;
        readyForConfirmation?: unknown;
      };
    }
    throw new Error('Malformed JSON output from voice analyze model');
  }
}

export async function analyzeVoiceServiceRequest(params: {
  message: string;
  history: VoiceTurnInput[];
  currentDraft: VoiceServiceDraft;
}): Promise<VoiceAnalyzeResult> {
  const ai = createGeminiClient(
    'عذرًا، خدمة الذكاء الاصطناعي غير مهيأة حاليًا. يمكنك استخدام نموذج الطلب مباشرة أو التواصل معنا عبر الدعم الفني.'
  );
  const safeCurrentDraft = sanitizeDraft(params.currentDraft);

  const systemInstruction = `أنت موظف خدمة عملاء صوتي محترف في «${SITE_INFO.fullNameAr}» (${SITE_INFO.serviceArea}).
تفهم العربية واللهجة اليمنية والخليجية وتحوّل كلام العميل لطلب منظم.

البيانات المجمّعة سابقًا:
${JSON.stringify(safeCurrentDraft)}

الخدمات الأربعة المعتمدة حرفيًا:
- "تركيب راوتر أو جهاز"
- "الدعم الفني"
- "صيانة وإصلاح الأعطال"
- "نقل الخدمة"

قيم الحقول الرسمية (اختر المطابق فقط أو اترك غير المذكور نصًا فارغًا ""):
- subscriberStatus: "نعم، لدي اشتراك حالي." | "لا، أريد الاشتراك لأول مرة." | "كنت مشتركاً سابقاً."
- deviceCategory: "أجهزة الراوتر والمودم (A)" | "أجهزة الربط واللاقِطات (B)"
- routerDevice: "A1 – ONE (TP-Link)" | "A2 – TWO (راوتر لاسلكي)" | "A3 – THREE (مودم KT)" | "A4 – FOUR (مودم LG)"
- linkDevice: "B1 – NanoStation" | "B2 – PowerBeam" | "B3 – LiteBeam" | "B4 – NanoStation M5"
- supportProblemType: "انقطاع الإنترنت بالكامل." | "بطء سرعة الإنترنت." | "ضعف إشارة الواي فاي." | "عدم القدرة على الاتصال بالشبكة." | "مشكلة في الراوتر أو المودم." | "مشكلة أخرى."

القواعد:
1. حافظ على البيانات السابقة وأضف أي معلومة جديدة فورًا في حقلها المناسب (fullName، phone، district، إلخ) ولا تضع الاسم والهاتف داخل details إذا وُجد حقلها.
2. اسأل عن النقص فقط في assistantMessage بجملة أو جملتين قصيرتين مناسبتين للاستماع، ولا تكرر ما أجاب عنه العميل.
3. اجعل readyForConfirmation = true بمجرد اتضاح الخدمة والمنطقة والتفاصيل الأساسية (أو عند تأكيد العميل)، وادعُه لمراجعة الملخص وتأكيده.
4. أخرج JSON صالحًا ومختصرًا بدون حشو أو تكرار.`;

  const contents = [
    ...params.history.slice(-10).map((turn) => ({
      role: turn.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: turn.content }],
    })),
    {
      role: 'user',
      parts: [{ text: params.message }],
    },
  ];

  const candidateModels = ['gemini-3.1-flash-lite', 'gemini-4-hydrogen'] as const;
  const availableModels = candidateModels.filter((m) => !isModelInCooldown(m));
  const modelsToTry =
    availableModels.length > 0 ? [...availableModels] : [...candidateModels];

  const requestDeadline = Date.now() + 16_000;
  let lastError: unknown;
  let lastStatus = 502;
  let sawDailyQuota = false;

  for (const modelName of modelsToTry) {
    const maxAttempts = modelName === 'gemini-3.1-flash-lite' ? 3 : 2;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      const remainingMs = requestDeadline - Date.now();
      if (remainingMs < 1500) break;

      const startedAt = Date.now();
      try {
        const response = await ai.models.generateContent({
          model: modelName,
          contents,
          config: {
            systemInstruction,
            maxOutputTokens: 2048,
            abortSignal: AbortSignal.timeout(Math.min(10_000, remainingMs)),
            ...(modelName === 'gemini-4-hydrogen'
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : {}),
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                assistantMessage: {
                  type: Type.STRING,
                  description: 'رد عربي قصير وودود (1-2 جملة).',
                },
                readyForConfirmation: {
                  type: Type.BOOLEAN,
                  description: 'true إذا كانت البيانات الأساسية جاهزة للمراجعة.',
                },
                serviceType: { type: Type.STRING },
                fullName: { type: Type.STRING },
                phone: { type: Type.STRING },
                district: { type: Type.STRING },
                subscriberStatus: { type: Type.STRING },
                deviceCategory: { type: Type.STRING },
                routerDevice: { type: Type.STRING },
                linkDevice: { type: Type.STRING },
                supportProblemType: { type: Type.STRING },
                maintenanceDeviceName: { type: Type.STRING },
                currentDistrict: { type: Type.STRING },
                newDistrict: { type: Type.STRING },
                details: { type: Type.STRING },
              },
              required: ['assistantMessage', 'readyForConfirmation', 'serviceType'],
              propertyOrdering: [
                'assistantMessage',
                'readyForConfirmation',
                'serviceType',
                'fullName',
                'phone',
                'district',
                'subscriberStatus',
                'deviceCategory',
                'routerDevice',
                'linkDevice',
                'supportProblemType',
                'maintenanceDeviceName',
                'currentDistrict',
                'newDistrict',
                'details',
              ],
            },
          },
        });

        const rawText = (response.text ?? '').trim();
        const finishReason = response.candidates?.[0]?.finishReason;
        if (!rawText) {
          throw new Error(
            `Empty response from ${modelName} (finishReason=${String(finishReason ?? 'unknown')})`
          );
        }

        const parsed = parseVoiceJson(rawText);

        const sanitizedIncoming = sanitizeDraft(parsed);
        const fallbackPhone = extractPhoneFallback(params.message);

        const mergedDraft: VoiceServiceDraft = {
          serviceType:
            sanitizedIncoming.serviceType || safeCurrentDraft.serviceType || '',
          fullName: sanitizedIncoming.fullName || safeCurrentDraft.fullName,
          phone:
            sanitizedIncoming.phone || safeCurrentDraft.phone || fallbackPhone,
          district: sanitizedIncoming.district || safeCurrentDraft.district,
          subscriberStatus:
            sanitizedIncoming.subscriberStatus ||
            safeCurrentDraft.subscriberStatus,
          deviceCategory:
            sanitizedIncoming.deviceCategory || safeCurrentDraft.deviceCategory,
          routerDevice:
            sanitizedIncoming.routerDevice || safeCurrentDraft.routerDevice,
          linkDevice:
            sanitizedIncoming.linkDevice || safeCurrentDraft.linkDevice,
          supportProblemType:
            sanitizedIncoming.supportProblemType ||
            safeCurrentDraft.supportProblemType,
          details: sanitizedIncoming.details || safeCurrentDraft.details,
          maintenanceDeviceName:
            sanitizedIncoming.maintenanceDeviceName ||
            safeCurrentDraft.maintenanceDeviceName,
          currentDistrict:
            sanitizedIncoming.currentDistrict ||
            safeCurrentDraft.currentDistrict,
          newDistrict:
            sanitizedIncoming.newDistrict || safeCurrentDraft.newDistrict,
        };

        const assistantMessage =
          typeof parsed.assistantMessage === 'string' &&
          parsed.assistantMessage.trim().length > 0
            ? parsed.assistantMessage.trim().slice(0, 600)
            : 'أهلاً بك، أخبرني بالخدمة التي تحتاجها وسأساعدك في تجهيز الطلب.';

        const missingFields = computeMissingFields(mergedDraft);
        const hasServiceType = Boolean(mergedDraft.serviceType);
        const readyForConfirmation =
          Boolean(parsed.readyForConfirmation) && hasServiceType;

        return {
          assistantMessage,
          updatedDraft: mergedDraft,
          missingFields,
          readyForConfirmation,
        };
      } catch (error: unknown) {
        if (error instanceof AssistantServiceError) throw error;
        lastError = error;
        const errObj = error as { status?: number; message?: string; name?: string };
        const status = errObj?.status;
        const errMsg = String(errObj?.message ?? error ?? '');
        const latencyMs = Date.now() - startedAt;

        if (
          status === 400 ||
          status === 401 ||
          status === 403 ||
          errMsg.includes('API_KEY_INVALID') ||
          errMsg.includes('PERMISSION_DENIED')
        ) {
          console.error(
            `[VoiceAnalyze] ${modelName} auth/request error (${status}, ${latencyMs}ms):`,
            errMsg
          );
          throw new AssistantServiceError(
            `Gemini auth/request error (${status}): ${errMsg}`,
            status === 401 ? 401 : status === 403 ? 403 : 503,
            'تعذر الاتصال بخدمة الذكاء الاصطناعي حاليًا بسبب إعدادات مفتاح API. يمكنك تعبئة نموذج الطلب مباشرة.'
          );
        }

        if (status === 404 || errMsg.includes('NOT_FOUND')) {
          markModelCooldown(modelName, 5 * 60_000);
          console.warn(
            `[VoiceAnalyze] ${modelName} unavailable (404, ${latencyMs}ms); switching to fallback model.`
          );
          break;
        }

        if (status === 429 || errMsg.includes('RESOURCE_EXHAUSTED')) {
          lastStatus = 429;
          const isDailyLimit =
            errMsg.includes('per day') || errMsg.includes('free_tier_requests');
          if (isDailyLimit) sawDailyQuota = true;
          markModelCooldown(modelName, isDailyLimit ? 10 * 60_000 : 30_000);
          console.warn(
            `[VoiceAnalyze] ${modelName} quota reached (429, ${latencyMs}ms); switching to fallback model.`
          );
          break;
        }

        if (
          errObj?.name === 'AbortError' ||
          errObj?.name === 'TimeoutError' ||
          errMsg.toLowerCase().includes('aborted') ||
          errMsg.toLowerCase().includes('timed out')
        ) {
          markModelCooldown(modelName, 20_000);
          console.warn(
            `[VoiceAnalyze] ${modelName} timed out on attempt ${attempt} (${latencyMs}ms); switching to fallback model.`
          );
          break;
        }

        if (status === 503 || status === 500 || errMsg.includes('UNAVAILABLE')) {
          lastStatus = 503;
          console.warn(
            `[VoiceAnalyze] ${modelName} transient ${status ?? '503'} on attempt ${attempt} (${latencyMs}ms).`
          );
        } else {
          console.error(
            `[VoiceAnalyze] ${modelName} attempt ${attempt} error (${status ?? errObj?.name ?? 'unknown'}, ${latencyMs}ms):`,
            errMsg
          );
        }
        if (attempt < maxAttempts) {
          const backoffMs = 350 * Math.pow(2, attempt - 1);
          const timeLeft = requestDeadline - Date.now();
          if (timeLeft > backoffMs + 1500) {
            await new Promise((r) => setTimeout(r, backoffMs));
          } else {
            break;
          }
        }
      }
    }
  }

  const msg = lastError instanceof Error ? lastError.message : String(lastError);
  if (lastStatus === 429 || sawDailyQuota) {
    throw new AssistantServiceError(
      'Gemini free-tier quota exceeded across candidate models (429).',
      429,
      'تشهد خدمة الذكاء الاصطناعي ضغطًا مؤقتًا حاليًا أو تم بلوغ حد الاستخدام. يمكنك المحاولة بعد لحظات أو تعبئة نموذج الطلب مباشرة.'
    );
  }
  throw new AssistantServiceError(
    `Voice analyze error (${lastStatus}): ${msg.slice(0, 180)}`,
    lastStatus === 503 ? 503 : 502,
    'تشهد خدمة الذكاء الاصطناعي ضغطًا مؤقتًا حاليًا. يرجى المحاولة مرة أخرى بعد لحظات أو تعبئة نموذج الخدمة مباشرة.'
  );
}

export async function synthesizeArabicSpeech(text: string): Promise<string> {
  const ai = createGeminiClient('خدمة النطق الصوتي غير مهيأة حاليًا.');
  const cleanText = text.trim().slice(0, 420);
  if (!cleanText) {
    throw new AssistantServiceError('Empty TTS text', 400, 'لا يوجد نص للقراءة.');
  }

  if (isModelInCooldown('gemini-3.8-flash-lite-tts')) {
    throw new AssistantServiceError(
      'TTS model temporarily in cooldown',
      503,
      'خدمة الصوت من الخادم غير متاحة حاليًا، يتم استخدام قارئ المتصفح كبديل.'
    );
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash-lite-tts',
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: cleanText,
              speechMetadata: {
                style: 'Warm, clear, polite Arabic customer service representative',
              },
            },
          ],
        },
      ],
      config: {
        abortSignal: AbortSignal.timeout(12_000),
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' },
          },
        },
      },
    });

    const base64Audio =
      response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) {
      throw new Error('No audio data returned by TTS model');
    }
    return base64Audio;
  } catch (error: unknown) {
    const errObj = error as { status?: number; message?: string };
    const status = errObj?.status;
    const msg = error instanceof Error ? error.message : String(error);

    if (status === 429 || msg.includes('RESOURCE_EXHAUSTED')) {
      markModelCooldown('gemini-3.8-flash-lite-tts', 30_000);
      console.warn('[VoiceTTS] quota reached (429); browser SpeechSynthesis fallback active.');
      throw new AssistantServiceError(
        `TTS quota exceeded (429): ${msg}`,
        429,
        'تم تجاوز حد الاستخدام المؤقت لنموذج الصوت، يتم استخدام قارئ المتصفح كبديل.'
      );
    }

    if (status === 400 || status === 404) {
      markModelCooldown('gemini-3.8-flash-lite-tts', 5 * 60_000);
      console.warn(`[VoiceTTS] model unavailable (${status}); using browser SpeechSynthesis fallback.`);
    } else {
      console.error(`[VoiceTTS] error (${status ?? 'unknown'}):`, msg);
    }

    throw new AssistantServiceError(
      `TTS error (${status ?? 502}): ${msg}`,
      status && status >= 400 && status < 500 ? status : 502,
      'تعذر تشغيل الرد الصوتي من الخادم حاليًا، يمكنك قراءة الرد نصيًا أو عبر قارئ المتصفح.'
    );
  }
}
