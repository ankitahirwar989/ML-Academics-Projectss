import subprocess
import sys
import os

def install_and_import():
    try:
        import fpdf
    except ImportError:
        print("Installing fpdf2...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
        import fpdf

install_and_import()
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "Indian Language Audio AI Updates", border=False, ln=True, align="C")
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

html_content = """
<h2>Objective</h2>
<p>To implement an audio preview capability in the Indian Language Audio classification model, allowing users to listen to their uploaded audio files before processing.</p>
<br>

<h2>Changes Implemented</h2>

<b>1. Frontend Audio Player Integration (audio.html)</b>
<ul>
  <li><b>Audio Element:</b> Inserted an HTML5 &lt;audio controls&gt; element below the drag-and-drop file upload zone. It is initially styled to be hidden.</li>
  <li><b>Responsive Design:</b> Styled the player to fit 100% of the container width with rounded borders to match the existing UI aesthetics.</li>
</ul>

<b>2. JavaScript Preview Logic</b>
<ul>
  <li><b>Blob URL Generation:</b> Leveraged <i>URL.createObjectURL()</i> in the file input's 'change' event listener to generate a temporary, browser-memory-hosted URL for the selected audio file.</li>
  <li><b>Dynamic Display:</b> Once the file is selected, the script maps the blob URL to the &lt;audio&gt; element's <i>src</i> attribute and reveals the audio player.</li>
  <li><b>Memory Management:</b> Added a window <i>beforeunload</i> event hook to safely invoke <i>URL.revokeObjectURL()</i> and discard the temporary memory blob when the user navigates away or refreshes the page, preventing browser memory leaks.</li>
</ul>

<b>3. Backend Status</b>
<ul>
  <li>No adjustments were necessary for the Flask backend (<i>app.py</i>) or the <i>/api/audio/predict</i> route, as the native file payload submission remained structurally unchanged.</li>
</ul>
"""

pdf.write_html(html_content)
pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audio_Model_Play_Feature_Updates.pdf")
pdf.output(pdf_path)
print(f"PDF successfully generated at: {pdf_path}")
