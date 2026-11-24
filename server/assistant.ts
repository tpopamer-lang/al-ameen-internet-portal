import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import {
  ASSISTANT_ACTIONS,
  VALID_ACTION_IDS,
  isValidActionId,
  type AssistantActionId,
} from '../src/config/assistantActions.ts';
import { DEVICES, DEVICES_NOTE } from '../src/config/devices.ts';
import {
  INTERNET_CARDS,
  MONTHLY_PACKAGES,
  SERVICE_OPTIONS,
} from '../src/config/forms.ts';
import { ROUTE_META, type AppRoutePath } from '../src/config/routes.ts';
import { ABOUT_CONTENT, SITE_INFO } from '../src/config/site.ts';

export interface ChatTurnInput {
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantReply {
  reply: string;
  actionIds: AssistantActionId[];
}

export class AssistantServiceError extends Error {
  public readonly statusCode: number;
  public readonly userMessage: string;

  constructor(message: string, statusCode: number, userMessage: string) {
    super(message);
    this.name = 'AssistantServiceError';
    this.statusCode = statusCode;
    this.userMessage = userMessage;
  }
}

// Shared per-model quota cooldown store so exhausted models are skipped smoothly
// instead of burning roundtrip latency and error logs on every user request.
const modelCooldownUntil = new Map<string, number>();
const DEFAULT_QUOTA_COOLDOWN_MS = 30_000;

export function isModelInCooldown(modelName: string): boolean {
  const until = modelCooldownUntil.get(modelName);
  if (!until) return false;
  if (Date.now() >= until) {
    modelCooldownUntil.delete(modelName);
    return false;
  }
  return true;
}

export function markModelCooldown(
  modelName: string,
  durationMs = DEFAULT_QUOTA_COOLDOWN_MS
): void {
  const nextUntil = Date.now() + durationMs;
  const existing = modelCooldownUntil.get(modelName) ?? 0;
  if (nextUntil > existing) {
    modelCooldownUntil.set(modelName, nextUntil);
  }
}

export function createGeminiClient(
  missingKeyMessage = 'عذرًا، خدمة المساعد الذكي غير مهيأة حاليًا. يمكنك التواصل معنا مباشرة عبر صفحة اتصل بنا أو الدعم الفني.'
): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    throw new AssistantServiceError(
      'GEMINI_API_KEY is not configured on the server.',
      503,
      missingKeyMessage
    );
  }

  return new GoogleGenAI({
    apiKey,
    apiVersion: 'v1beta',
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

const ROUTE_LABELS: Record<AppRoutePath, string> = {
  '/': 'الصفحة الرئيسية',
  '/about': 'من نحن',
  '/services': 'خدماتنا والأجهزة المتاحة',
  '/internet-cards': 'كروت الإنترنت والباقات الشهرية',
  '/request': 'طلب خدمة أو اشتراك',
  '/support': 'الدعم الفني',
  '/contact': 'اتصل بنا',
};

function buildSystemInstruction(currentPath: AppRoutePath): string {
  const cardsText = INTERNET_CARDS.map(
    (c) => `- ${c.title} (${c.code}): السعر ${c.price}، السعة ${c.capacity}`
  ).join('\n');

  const packagesText = MONTHLY_PACKAGES.map(
    (p) => `- ${p.title}: السعر ${p.price} شهريًا، السعة ${p.capacity}`
  ).join('\n');

  const servicesText = SERVICE_OPTIONS.map(
    (s) => `- ${s.title}: ${s.description}`
  ).join('\n');

  const devicesText = DEVICES.map(
    (d) => `- ${d.numberLabel} | ${d.name} (${d.type}): ${d.description}`
  ).join('\n');

  const pagesText = (Object.keys(ROUTE_META) as AppRoutePath[])
    .map((path) => `- ${path} (${ROUTE_LABELS[path]}): ${ROUTE_META[path].description}`)
    .join('\n');

  const actionsCatalogText = VALID_ACTION_IDS.map(
    (id) => `- "${id}": ${ASSISTANT_ACTIONS[id].label}`
  ).join('\n');

  return `أنت «مساعد الأمين الذكي»، موظف خدمة العملاء الرقمي الرسمي لموقع «${SITE_INFO.fullNameAr}» (${SITE_INFO.fullNameEn}).

شخصيتك ومهمتك:
- اسمك الرسمي: «مساعد الأمين الذكي» (ويُشار إليك أيضًا بـ «مساعد الأمين»).
- تساعد زوار بوابة الأمين للإنترنت بأسلوب عربي مهذب، احترافي، واضح، وعملي.
- تجعل الرد مباشرًا ومفيدًا ومختصرًا (1 إلى 4 جمل أو نقاط واضحة دون إطالة).

سياق الزائر الحالي:
- العميل يتصفح حاليًا: ${ROUTE_LABELS[currentPath]} (${currentPath}).
- استفد من الصفحة الحالية عندما يطرح سؤالًا عامًا مثل "كم الأسعار؟" أو "كيف أطلب؟" دون فرض الصفحة عليه إذا سأل عن قسم آخر.

القواعد الصارمة (يجب الالتزام بها دائمًا):
1. مصدرك الأساسي والوحيد للمعلومات هو بيانات الموقع المرفقة أدناه فقط.
2. يُمنع منعًا باتًا اختراع أو تخمين أي أسعار، باقات، سعات، سرعات غير مذكورة، مواعيد عمل، عروض، أرقام هواتف، روابط، أو خدمات غير موجودة في بيانات الموقع.
3. إذا سأل العميل عن معلومة غير موجودة في بيانات الموقع (مثل: فحص رصيد كرت معيّن، حالة برج أو تغطية لحظية، أسعار الأجهزة بالريال، أو مواضيع خارج اختصاص بوابة الأمين):
   - وضّح له بلطف وبشكل صريح أن هذه المعلومة غير متوفرة لديك عبر البوابة الإلكترونية.
   - وجّهه مباشرة إلى الدعم الفني أو التواصل عبر الهاتف (779577209) أو واتساب.
4. افهم الأسئلة حتى لو كانت مكتوبة بالعامية أو اللهجة اليمنية والخليجية أو بها أخطاء إملائية (مثل: "كم سعر النت", "ابغى اركب راوتر", "النت خربان", "كم كرت ابو 500", "وش الباقات عندكم", "وين مكانكم")، وأجب بعربية واضحة وسلسة.
5. احفظ سياق المحادثة بالكامل ولا تطلب من العميل أبدًا تكرار معلومة ذكرها سابقًا في المحادثة (وتابع الأسئلة المتصلة مثل: "طيب واللي أكبر منه؟"، "كم سعرها؟"، "كيف أطلبها؟").
6. لا تدّعِ أبدًا أنك أرسلت طلبًا أو فتحت نموذجًا نيابةً عنه؛ وضّح له الطريقة وأرفق أزرار التوجيه المناسبة.
7. وجّه العميل بوضوح لخطوات الطلب الفعلية في الموقع (صفحة طلب خدمة / نموذج Google الرسمي / أو ميزة اطلب خدمتك بصوتك عند الحاجة).

توجيه العميل عبر الأزرار التفاعلية (suggestedActionIds):
اختر من 0 إلى 3 معرفات فقط من القائمة أدناه (ولا تخترع أي معرف غير موجود):
- كروت الإنترنت أو الأسعار أو السعات: "page_internet_cards" (و"form_subscriptions" إذا أراد الطلب).
- الباقات الشهرية أو الاشتراكات: "page_internet_cards" و"form_subscriptions" (أو "page_request").
- تركيب راوتر أو جهاز أو معرفة الأجهزة: "page_services" و"form_install_device" (أو "page_request").
- صيانة أو إصلاح أعطال: "form_maintenance" و"page_services".
- نقل الخدمة: "form_transfer" و"page_services".
- مشكلة فنية أو الدعم الفني: "page_support" و"form_support".
- طلب خدمة عامة أو اختيار نموذج: "page_request" و"form_services".
- التواصل أو الأرقام أو العنوان: "page_contact" و"contact_whatsapp" و"contact_phone" (أو "contact_maps" للموقع الجغرافي، أو "contact_email" للبريد).
- نبذة عن الشبكة: "page_about".

معرفات الإجراءات المتاحة:
${actionsCatalogText}

==============================
بيانات موقع بوابة الأمين للإنترنت الرسمية:
==============================

[معلومات عامة والموقع]
- الاسم العربي: ${SITE_INFO.fullNameAr}
- الاسم الإنجليزي: ${SITE_INFO.fullNameEn}
- الشعار: ${SITE_INFO.tagline}
- طبيعة النشاط: ${SITE_INFO.activityNature}
- نطاق الخدمة: ${SITE_INFO.serviceArea}
- العنوان ومقر الشبكة: ${SITE_INFO.addressDisplay}

[من نحن]
- التعريف: ${ABOUT_CONTENT.intro}
- الرؤية: ${ABOUT_CONTENT.visionText}
- الرسالة: ${ABOUT_CONTENT.missionText}
- أهداف البوابة:
${ABOUT_CONTENT.goals.map((g) => `- ${g}`).join('\n')}

[وسائل التواصل الرسمية]
- رقم الاتصال الهاتفي: 779577209
- رقم واتساب: 967779577209
- البريد الإلكتروني: omerbinhmood@gmail.com
- الموقع على الخرائط: ${SITE_INFO.address}

[صفحات الموقع]
${pagesText}

[الخدمات الإلكترونية المتاحة]
${servicesText}

[صفحة طلب خدمة أو اشتراك (/request)]
تضم منشئ طلب إلكتروني وتحتوي على نموذجي Google Forms الرسميين:
1. نموذج طلب خدمة: لطلبات التركيب أو الصيانة أو الدعم الفني أو نقل الخدمة.
2. نموذج اشتراك أو كروت إنترنت: لطلب كروت الإنترنت اليومية أو الباقات الشهرية.
كما تتوفر ميزة «اطلب خدمتك بصوتك» لتجهيز طلبات الخدمة بالصوت.

[صفحة الدعم الفني (/support)]
تتيح تشخيصًا أوليًا سريعًا للمشكلة وإرسال بلاغ فني مع تحديد نوع الخدمة (الدعم الفني) تلقائيًا، بالإضافة إلى الاتصال وواتساب.

[كروت الإنترنت المعتمدة (/internet-cards)]
${cardsText}

[الباقات الشهرية المعتمدة (/internet-cards)]
${packagesText}

[الأجهزة المتاحة للتركيب والربط (/services)]
${devicesText}
${DEVICES_NOTE}
ملاحظة: الموقع يعرض مواصفات عامة للأجهزة دون إدراج أسعار لها؛ إذا سأل العميل عن سعر جهاز معين فاذكر له بلطف أن أسعار الأجهزة غير مدرجة على الموقع ووجّهه للتواصل مع خدمة العملاء.`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseAssistantJson(rawText: string): {
  reply?: unknown;
  suggestedActionIds?: unknown;
} {
  let clean = rawText.trim();
  if (clean.startsWith('```')) {
    clean = clean
      .replace(/^```(?:json)?\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim();
  }

  try {
    return JSON.parse(clean) as {
      reply?: unknown;
      suggestedActionIds?: unknown;
    };
  } catch {
    const startIdx = clean.indexOf('{');
    const endIdx = clean.lastIndexOf('}');
    if (startIdx >= 0 && endIdx > startIdx) {
      return JSON.parse(clean.slice(startIdx, endIdx + 1)) as {
        reply?: unknown;
        suggestedActionIds?: unknown;
      };
    }
    if (!clean.startsWith('{') && clean.length > 0) {
      return { reply: clean.slice(0, 1200), suggestedActionIds: [] };
    }
    throw new Error('Malformed JSON output from assistant model');
  }
}

export async function generateAssistantReply(params: {
  message: string;
  history: ChatTurnInput[];
  currentPath: AppRoutePath;
}): Promise<AssistantReply> {
  const ai = createGeminiClient();

  const contents = [
    ...params.history.map((turn) => ({
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
            systemInstruction: buildSystemInstruction(params.currentPath),
            maxOutputTokens: 2048,
            abortSignal: AbortSignal.timeout(Math.min(10_000, remainingMs)),
            ...(modelName === 'gemini-4-hydrogen'
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : {}),
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                reply: {
                  type: Type.STRING,
                  description:
                    'الرد العربي المختصر والواضح والمفيد للعميل بالاعتماد على بيانات الموقع فقط.',
                },
                suggestedActionIds: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.STRING,
                  },
                  description:
                    'قائمة بمعرفات الإجراءات المرتبطة بالرد من القائمة المسموح بها فقط (0 إلى 3 معرفات).',
                },
              },
              required: ['reply', 'suggestedActionIds'],
              propertyOrdering: ['reply', 'suggestedActionIds'],
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

        const parsed = parseAssistantJson(rawText);

        const replyText =
          typeof parsed.reply === 'string' && parsed.reply.trim().length > 0
            ? parsed.reply.trim()
            : 'مرحبًا بك في بوابة الأمين، كيف يمكنني مساعدتك اليوم؟';

        const rawIds = Array.isArray(parsed.suggestedActionIds)
          ? parsed.suggestedActionIds
          : [];

        const uniqueValidIds: AssistantActionId[] = [];
        for (const item of rawIds) {
          if (
            typeof item === 'string' &&
            isValidActionId(item) &&
            !uniqueValidIds.includes(item)
          ) {
            uniqueValidIds.push(item);
            if (uniqueValidIds.length >= 3) break;
          }
        }

        return {
          reply: replyText,
          actionIds: uniqueValidIds,
        };
      } catch (error: unknown) {
        lastError = error;
        const errObj = error as { status?: number; message?: string; name?: string };
        const status = errObj?.status;
        const msg = String(errObj?.message ?? error ?? '');
        const latencyMs = Date.now() - startedAt;

        if (
          status === 400 ||
          status === 401 ||
          status === 403 ||
          msg.includes('API_KEY_INVALID') ||
          msg.includes('PERMISSION_DENIED')
        ) {
          console.error(
            `[AlAmeenAssistant] ${modelName} auth/request error (${status}, ${latencyMs}ms):`,
            msg
          );
          throw new AssistantServiceError(
            `Gemini auth/request error (${status}): ${msg}`,
            status === 401 ? 401 : status === 403 ? 403 : 503,
            'تعذر الاتصال بمساعد الأمين الذكي حاليًا بسبب إعدادات الخدمة. يرجى التواصل معنا عبر صفحة اتصل بنا أو الدعم الفني.'
          );
        }

        if (status === 404 || msg.includes('NOT_FOUND')) {
          markModelCooldown(modelName, 5 * 60_000);
          console.warn(
            `[AlAmeenAssistant] ${modelName} unavailable (404, ${latencyMs}ms); switching to fallback model.`
          );
          break;
        }

        if (status === 429 || msg.includes('RESOURCE_EXHAUSTED')) {
          lastStatus = 429;
          const isDailyLimit = msg.includes('per day') || msg.includes('free_tier_requests');
          if (isDailyLimit) sawDailyQuota = true;
          markModelCooldown(
            modelName,
            isDailyLimit ? 10 * 60_000 : DEFAULT_QUOTA_COOLDOWN_MS
          );
          console.warn(
            `[AlAmeenAssistant] ${modelName} quota reached (429, ${latencyMs}ms); switching to fallback model.`
          );
          break;
        }

        if (
          errObj?.name === 'AbortError' ||
          errObj?.name === 'TimeoutError' ||
          msg.toLowerCase().includes('aborted') ||
          msg.toLowerCase().includes('timed out')
        ) {
          markModelCooldown(modelName, 20_000);
          console.warn(
            `[AlAmeenAssistant] ${modelName} timed out on attempt ${attempt} (${latencyMs}ms); switching to fallback model.`
          );
          break;
        }

        if (status === 503 || status === 500 || msg.includes('UNAVAILABLE')) {
          lastStatus = 503;
          console.warn(
            `[AlAmeenAssistant] ${modelName} transient ${status ?? '503'} on attempt ${attempt} (${latencyMs}ms).`
          );
        } else {
          console.error(
            `[AlAmeenAssistant] ${modelName} attempt ${attempt} error (${status ?? errObj?.name ?? 'unknown'}, ${latencyMs}ms):`,
            msg
          );
        }
        if (attempt < maxAttempts) {
          const backoffMs = 350 * Math.pow(2, attempt - 1);
          const timeLeft = requestDeadline - Date.now();
          if (timeLeft > backoffMs + 1500) {
            await sleep(backoffMs);
          } else {
            break;
          }
        }
      }
    }
  }

  const fallbackMsg = lastError instanceof Error ? lastError.message : String(lastError);
  if (lastStatus === 429 || sawDailyQuota) {
    throw new AssistantServiceError(
      'Gemini free-tier quota exceeded across candidate models (429).',
      429,
      'مساعد الأمين الذكي يشهد ضغطًا مرتفعًا حاليًا أو تم بلوغ حد الاستخدام المؤقت. يمكنك المحاولة بعد قليل أو استخدام صفحات الموقع وقنوات التواصل المباشر.'
    );
  }
  throw new AssistantServiceError(
    `Gemini upstream failure (${lastStatus}): ${fallbackMsg.slice(0, 180)}`,
    lastStatus === 503 ? 503 : 502,
    'تشهد خدمة الذكاء الاصطناعي ضغطًا مؤقتًا حاليًا. يرجى المحاولة مرة أخرى بعد لحظات أو التواصل عبر صفحة اتصل بنا.'
  );
}
