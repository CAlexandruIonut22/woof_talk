const OPENAI_API_URL = "https://api.openai.com/v1/responses";

const CANINE_INTERPRETATION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "likelyMeaning",
    "humanPhrase",
    "confidenceLevel",
    "confidenceScore",
    "possibleMeanings",
    "why",
    "recommendedAction",
    "warnings",
    "disclaimer"
  ],
  properties: {
    likelyMeaning: {
      type: "string",
      description: "Short description of the most likely communication intent."
    },
    humanPhrase: {
      type: "string",
      description: "A careful human-style phrase, not a literal translation."
    },
    confidenceLevel: {
      type: "string",
      enum: ["low", "medium", "high"]
    },
    confidenceScore: {
      type: "number",
      minimum: 0,
      maximum: 1
    },
    possibleMeanings: {
      type: "array",
      minItems: 2,
      maxItems: 4,
      items: {
        type: "object",
        additionalProperties: false,
        required: ["label", "probability", "reason"],
        properties: {
          label: { type: "string" },
          probability: { type: "number", minimum: 0, maximum: 1 },
          reason: { type: "string" }
        }
      }
    },
    why: {
      type: "array",
      minItems: 2,
      maxItems: 5,
      items: { type: "string" }
    },
    recommendedAction: {
      type: "string"
    },
    warnings: {
      type: "array",
      maxItems: 4,
      items: { type: "string" }
    },
    disclaimer: {
      type: "string"
    }
  }
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8"
    }
  });
}

function extractOutputText(openAiResponse) {
  if (typeof openAiResponse.output_text === "string") {
    return openAiResponse.output_text;
  }

  const textParts = [];

  for (const item of openAiResponse.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) {
        textParts.push(content.text);
      }
    }
  }

  return textParts.join("\n").trim();
}

function buildPrompt(payload) {
  return `
You are Woof Talk, an AI assistant for interpreting dog vocalizations.

You must follow these rules:
- Do NOT claim literal dog-speech translation.
- Interpret probabilistically using the acoustic analysis, emotion prediction, context ranking, dog profile, and owner scene context.
- Use simple language for normal dog owners.
- Be useful: explain likely meaning, why, and what the owner should do next.
- Never diagnose medical conditions.
- If there are signs of pain, unusual sudden behavior, breathing distress, collapse, repeated distress sounds, or owner selected a pain-related scene, recommend contacting a veterinarian.
- If the recording quality is poor, lower confidence and say so.
- Keep the result concise but helpful.

Input data:
${JSON.stringify(payload, null, 2)}
`;
}

export default async function handler(request) {
  if (request.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    return jsonResponse(
      {
        error: "Missing OPENAI_API_KEY environment variable."
      },
      500
    );
  }

  let payload;

  try {
    payload = await request.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body." }, 400);
  }

  if (!payload || typeof payload !== "object") {
    return jsonResponse({ error: "Missing analysis payload." }, 400);
  }

  const prompt = buildPrompt(payload);

  try {
    const openAiResponse = await fetch(OPENAI_API_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: process.env.OPENAI_MODEL || "gpt-4o-mini",
        input: [
          {
            role: "user",
            content: [
              {
                type: "input_text",
                text: prompt
              }
            ]
          }
        ],
        text: {
          format: {
            type: "json_schema",
            name: "canine_interpretation",
            strict: true,
            schema: CANINE_INTERPRETATION_SCHEMA
          }
        },
        max_output_tokens: 700
      })
    });

    const data = await openAiResponse.json();

    if (!openAiResponse.ok) {
      return jsonResponse(
        {
          error: "OpenAI request failed.",
          details: data
        },
        openAiResponse.status
      );
    }

    const outputText = extractOutputText(data);
    const parsed = JSON.parse(outputText);

    return jsonResponse({
      source: "openai",
      model: data.model,
      ...parsed
    });
  } catch (error) {
    return jsonResponse(
      {
        error: "AI interpretation failed.",
        details: error?.message || String(error)
      },
      500
    );
  }
}