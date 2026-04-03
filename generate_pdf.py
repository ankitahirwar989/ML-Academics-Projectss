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
        self.cell(0, 10, "Face Mask Detection Updates", border=False, ln=True, align="C")
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

html_content = """
<h2>Objective</h2>
<p>To update the Face Mask Detection functionality from static image uploads to a real-time, live webcam stream.</p>
<br>

<h2>Changes Implemented</h2>

<b>1. Frontend UI Revamp (facemask.html)</b>
<ul>
  <li><b>Removed Image Upload:</b> The drag-and-drop zone and the file input for static image uploads were entirely removed.</li>
  <li><b>Added Live Camera UI:</b> Introduced an HTML5 &lt;video&gt; element to display the live feed from the user's webcam and a hidden &lt;canvas&gt; to capture snapshot frames.</li>
  <li><b>Camera Controls:</b> Added "Start Camera" and "Stop Camera" buttons to manage media stream access, allowing users to start and stop the live capture at their convenience.</li>
</ul>

<b>2. JavaScript Tracking Logic (facemask.html)</b>
<ul>
  <li><b>Camera Access:</b> Leveraged <i>navigator.mediaDevices.getUserMedia()</i> to request and render the video stream from the user's device.</li>
  <li><b>Continuous Polling:</b> Set up an interval loop that triggers every 800 milliseconds while the camera is active.</li>
  <li><b>Frame Capture & API Submission:</b> Inside the loop, the current frame of the &lt;video&gt; is drawn onto the hidden &lt;canvas&gt;, converted to a JPEG Blob, and appended to a FormData object with the name 'image'.</li>
  <li><b>Asynchronous Updates:</b> The frames are sequentially posted via <i>fetch()</i> to the backend. The HTML view is interactively updated per frame with the model's confidence distribution.</li>
</ul>

<b>3. Backend Compatibility (app.py)</b>
<ul>
  <li>The original architecture of the <i>/api/facemask/predict</i> route expected a file payload named 'image', so the frontend Blob formulation flawlessly matched the backend expectation.</li>
  <li>No adjustments were necessary for the Flask backend, maintaining its robust predictive logic seamlessly with live streaming elements.</li>
</ul>
"""

pdf.write_html(html_content)
pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Face_Mask_Live_Cam_Updates.pdf")
pdf.output(pdf_path)
print(f"PDF successfully generated at: {pdf_path}")
