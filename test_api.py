import requests

API_KEY  = "sk-ixbTrQOn0gQCP5FHF6cxHCXBOLlSZRoGuXMVo6QNJKy3PErn"
BASE_URL = "https://chat.cloudapi.vip/v1" # 注意末尾是 /v1，且不要带斜杠

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

# ---------- 1) 测试 Chat 接口 ----------
print("=" * 50)
print("① 测试 Chat (gpt-5)")
print("=" * 50)
r = requests.post(
    f"{BASE_URL}/chat/completions",
    headers=headers,
    json={
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "只回答两个字：你好"}],
    },
    timeout=30,
)
print("状态码:", r.status_code)
if r.status_code == 200:
    print("✅ 成功，回复：", r.json()["choices"][0]["message"]["content"])
else:
    print("❌ 失败，返回：", r.text)

# ---------- 2) 测试 Embeddings 接口 ----------
print()
print("=" * 50)
print("② 测试 Embeddings (text-embedding-ada-002)")
print("=" * 50)
r = requests.post(
    f"{BASE_URL}/embeddings",
    headers=headers,
    json={
        "model": "text-embedding-ada-002",
        "input": "hello world",
    },
    timeout=30,
)
print("状态码:", r.status_code)
if r.status_code == 200:
    vec = r.json()["data"][0]["embedding"]
    print(f"✅ 成功，向量维度 = {len(vec)}，前 5 个值 = {vec[:5]}")
else:
    print("❌ 失败，返回：", r.text)