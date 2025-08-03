const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const resultDiv = document.getElementById('result');
const uploadBtn = document.getElementById('uploadBtn');

fileInput.addEventListener('change', function() {
    const file = fileInput.files[0];
    if (!file) return;

    // Show image preview
    const reader = new FileReader();
    reader.onload = e => {
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Clear old results
    resultDiv.innerHTML = '';
});

uploadBtn.addEventListener('click', async function() {
    const file = fileInput.files[0];
    if (!file) {
        resultDiv.innerHTML = "Please select an image!";
        return;
    }

    resultDiv.innerHTML = "⏳ Classifying...";
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            resultDiv.innerHTML = "Server error. Please try another image.";
            return;
        }

        const data = await res.json();
        if (data.error) {
            resultDiv.innerHTML = data.error;
            return;
        }
        if (data.faces && data.faces.length > 0) {
            let mainEmotion = data.primary_emotion || data.faces[0].emotion;
            let msg = `<span style="font-size:1.4em;">😃</span> <b>Emotion:</b> <span style="color:#356aff;font-size:1.1em">${mainEmotion}</span>`;
            if (data.faces.length > 1) {
                msg += `<br>Detected <b>${data.faces.length}</b> faces.`;
            }
            // Optionally, list all faces/emotions
            msg += '<ul style="margin-top:8px;text-align:left;">';
            data.faces.forEach((f, i) => {
                msg += `<li>Face ${i+1}: <b>${f.emotion}</b></li>`;
            });
            msg += '</ul>';
            resultDiv.innerHTML = msg;
        } else {
            resultDiv.innerHTML = "No face detected in the image.";
        }
    } catch (err) {
        resultDiv.innerHTML = "❌ Could not reach backend. Check if API is running.";
    }
});