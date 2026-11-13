export async function request(path, options = {}) {
  const response = await fetch(`/api/${path}`, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid request. Please check your image and settings.');
  return data;
}
