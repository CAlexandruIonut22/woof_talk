export async function requestAiInterpretation(payload) {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: {
      "content-type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.error || "AI interpretation failed.");
  }

  return data;
}