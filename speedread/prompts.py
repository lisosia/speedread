PROMPTS = {
    "general": (
        "Transcribe the text of the book’s main body from the image. "
        "Ignore any unreadable parts, and output only the transcribed text."
    ),
    "ja": "画像内の,本の本文のテキストを書き起こしして. 読めない部分は無視して, 書き起こし文のみ出力して.",
    "ja_vert": "画像内の,本の本文のテキスト(縦書き)を書き起こしして. 読めない部分は無視して, 書き起こし文のみ出力して.",
}


def get_prompt(key: str) -> str:
    return PROMPTS.get(key, PROMPTS["general"])
