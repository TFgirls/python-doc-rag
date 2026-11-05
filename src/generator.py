from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def generate(query, context):
    """把检索到的文档片段作为上下文，调用 DeepSeek chat 生成答案（强制带引用来源）。"""
    if not DEEPSEEK_API_KEY:
        return "⚠️ 未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 中填入你的 DeepSeek API Key 后重试。"

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=30)
    prompt = (
        "你是文档智能问答助手。以下是为回答用户问题检索到的文档片段，"
        "请只依据这些片段作答；若片段中无对应信息，请明确说明“文档中未找到相关信息”。\n\n"
        f"【检索到的文档片段】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        "请用简洁中文回答，引用片段时标注对应文件名（如 [来源:xxx.txt]），并在末尾汇总所依据的文件。"
        "若片段中没有对应信息，只回一句“文档中未找到相关信息。”即可，不要解释、不要罗列片段、不要汇总文件。"
    )
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except (APITimeoutError, RateLimitError, APIError) as e:
        return f"⚠️ 调用 DeepSeek 失败（{type(e).__name__}），请检查网络或配额后重试。"
