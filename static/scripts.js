const predictButton = document.getElementById("predict-button");
const submitButton = document.getElementById("submit-button");
const predictedLabelDisplay = document.getElementById("predicted-label");
const notification = document.getElementById("notification");
const loadingSpinner = document.getElementById("loading-spinner");

let issueId = null;

const showNotification = (message) => {
    notification.innerText = message;
    notification.style.display = "block";

    setTimeout(() => {
        notification.style.display = "none";
    }, 5000);
};

predictButton.addEventListener("click", async () => {
    const title = document.getElementById("title").value;
    const body = document.getElementById("body").value;

    loadingSpinner.style.display = "block";
    predictButton.disabled = true;

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ title, body }),
        });

        const data = await response.json();
        issueId = data.id;
        predictedLabelDisplay.innerText = data.label;
        submitButton.disabled = false;
    } catch (error) {
        console.error("Error during prediction:", error);
        predictedLabelDisplay.innerText = "Error predicting label";
    } finally {
        loadingSpinner.style.display = "none";
        predictButton.disabled = false;
    }
});

submitButton.addEventListener("click", async () => {
    const selectedLabel = document.querySelector('input[name="label"]:checked').value;

    const response = await fetch("/api/correct", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ id: issueId, label: selectedLabel }),
    });

    const data = await response.json();
    showNotification("Correction submitted successfully!");
});
