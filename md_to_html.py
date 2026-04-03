import markdown
import sys

md_file = sys.argv[1]
html_file = sys.argv[2]

with open(md_file, 'r', encoding='utf-8') as f:
    text = f.read()

html = markdown.markdown(text, extensions=['tables', 'fenced_code'])

full_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 30px; color: #333; }}
    h1 {{ color: #1a73e8; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
    h2 {{ color: #d93025; margin-top: 30px; }}
    h3 {{ color: #188038; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background-color: #f8f9fa; font-weight: bold; border-bottom: 2px solid #ccc; }}
    code {{ background-color: #f1f3f4; padding: 2px 6px; border-radius: 4px; font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 90%; color: #d63384; }}
    pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 6px; border: 1px solid #e9ecef; overflow-x: auto; }}
    pre code {{ background-color: transparent; color: #212529; padding: 0; }}
    blockquote {{ border-left: 4px solid #1a73e8; margin: 20px 0; padding: 15px 20px; background-color: #f0f7ff; border-radius: 0 8px 8px 0; color: #555; }}
    hr {{ border: 0; border-top: 1px solid #eee; margin: 30px 0; }}
</style>
</head>
<body>
{html}
</body>
</html>
"""

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Created {html_file}")
