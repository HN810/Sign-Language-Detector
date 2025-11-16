// Frontend app: captures webcam frames and optionally sends them to a local backend /predict
// Usage: open index.html with Live Server (or any static server). To enable predictions
// run the optional Flask server included in frontend/server.py and toggle "Use Local Backend".

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const translateBtn = document.getElementById('translateBtn');
const snapBtn = document.getElementById('snapBtn');
const backendToggle = document.getElementById('backendToggle');
const resultText = document.getElementById('resultText');
const labelBadge = document.getElementById('labelBadge');

let streaming = false;
let videoWidth = 640, videoHeight = 480;

async function startCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280}, height:{ideal:720}}, audio:false});
    video.srcObject = stream;
    await video.play();
    streaming = true;
    videoWidth = video.videoWidth || videoWidth;
    videoHeight = video.videoHeight || videoHeight;
    overlay.width = videoWidth;
    overlay.height = videoHeight;
    requestAnimationFrame(drawLoop);
  }catch(err){
    labelBadge.textContent = 'Camera error';
    console.error('Could not start camera', err);
  }
}

function drawLoop(){
  // simple overlay: nothing fancy but reserved for future
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0,0,overlay.width, overlay.height);
  requestAnimationFrame(drawLoop);
}

function dataURLtoBlob(dataurl){
  const arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1], bstr = atob(arr[1]);
  let n = bstr.length; const u8arr = new Uint8Array(n);
  while(n--){ u8arr[n]=bstr.charCodeAt(n); }
  return new Blob([u8arr], {type:mime});
}

async function captureFrame(){
  if(!streaming) return null;
  const canvas = document.createElement('canvas');
  canvas.width = videoWidth; canvas.height = videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.9);
}

translateBtn.addEventListener('click', async ()=>{
  labelBadge.textContent = 'Capturing...';
  const dataUrl = await captureFrame();
  if(!dataUrl){ labelBadge.textContent='No video'; return; }

  if(backendToggle.checked){
    // call local backend
    labelBadge.textContent = 'Calling backend...';
    try{
      const r = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({image: dataUrl})
      });
      if(!r.ok) throw new Error('server returned '+r.status);
      const j = await r.json();
      if(j.status === 'ok'){
        resultText.textContent = j.label || j.prediction || '—';
        labelBadge.textContent = 'Result';
      }else if(j.status === 'no_hand'){
        resultText.textContent = 'No hand detected';
        labelBadge.textContent = 'No hand';
      }else{
        resultText.textContent = 'Error';
        labelBadge.textContent = 'Error';
      }
    }catch(err){
      console.error(err);
      labelBadge.textContent = 'Backend unreachable';
      resultText.textContent = 'Backend not running';
      // optionally fall back to demo
    }
  }else{
    // Demo mode: no backend. Show placeholder or random
    labelBadge.textContent = 'Demo mode';
    resultText.textContent = 'Demo — run backend to get real predictions';
  }
});

snapBtn.addEventListener('click', async ()=>{
  const dataUrl = await captureFrame();
  if(!dataUrl) return;
  const a = document.createElement('a');
  a.href = dataUrl; a.download = 'snapshot.jpg';
  document.body.appendChild(a); a.click(); a.remove();
});

// start camera on load
startCamera();

// allow pressing Enter key to perform Translate (for convenience)
document.addEventListener('keydown', (e)=>{
  if(e.key === 'Enter') translateBtn.click();
});
