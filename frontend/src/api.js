const apiBase = import.meta.env.VITE_API_BASE || "";

async function parseError(res) {
  try {
    const body = await res.json();
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (detail && detail.message) return detail.message;
    return res.statusText;
  } catch {
    return res.statusText;
  }
}

export async function analyzeImage(file, userId) {
  const form = new FormData();
  form.append("file", file);
  if (userId) {
    form.append("user_id", userId);
  }
  const res = await fetch(`${apiBase}/api/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchHistory(userId) {
  const url = new URL(`${apiBase}/api/analyses`);
  if (userId) {
    url.searchParams.set("user_id", userId);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchAnalysis(id, userId) {
  const url = new URL(`${apiBase}/api/analyses/${id}`);
  if (userId) {
    url.searchParams.set("user_id", userId);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export { apiBase };
