document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // UI Elements
    const btnText = document.getElementById('btn-text');
    const loader = document.getElementById('loader');
    const submitBtn = document.getElementById('submit-btn');
    const resContainer = document.getElementById('result-container');
    const resTitle = document.getElementById('result-title');
    const resProb = document.getElementById('result-prob');
    
    // Set loading state
    btnText.style.display = 'none';
    loader.style.display = 'block';
    submitBtn.disabled = true;
    resContainer.classList.remove('active', 'churn', 'retained');
    
    // Gather form data
    const formData = new FormData(this);
    const data = {};
    for (let [key, value] of formData.entries()) {
        if (key === 'SeniorCitizen' || key === 'tenure') {
            data[key] = parseInt(value, 10);
        } else if (key === 'MonthlyCharges') {
            data[key] = parseFloat(value);
        } else {
            data[key] = value; // TotalCharges left as string since backend expects it, or it will handle it.
        }
    }
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Something went wrong');
        }
        
        // Display results
        resContainer.classList.add('active');
        if (result.prediction === 'Churn') {
            resContainer.classList.add('churn');
            resTitle.innerText = "High Risk of Churn 🚨";
        } else {
            resContainer.classList.add('retained');
            resTitle.innerText = "Customer Retained ✅";
        }
        
        const probPerc = (result.probability * 100).toFixed(1);
        resProb.innerHTML = `Churn Probability: <strong>${probPerc}%</strong>`;
        
    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        // Reset loading state
        btnText.style.display = 'inline';
        loader.style.display = 'none';
        submitBtn.disabled = false;
    }
});
