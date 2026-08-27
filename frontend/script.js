const API_BASE_URL = "http://127.0.0.1:8000";

const emailText = document.querySelector("#emailText");
const count = document.querySelector("#count");
const button = document.querySelector("#analyzeBtn");
const error = document.querySelector("#error");
const result = document.querySelector("#result");

const examples = {
  safe: `Subject: Project update

Hi team, the weekly project meeting has been moved to Thursday at 3 PM. Please review the attached agenda before the meeting. Thanks!`,

  spam: `CONGRATULATIONS! You have been selected to receive a FREE cash reward. Click the link now and provide your bank details to claim before this offer expires!`,
};

emailText.addEventListener("input", () => {
  count.textContent = `${emailText.value.length.toLocaleString()} / 10,000`;
});

document.querySelectorAll("[data-sample]").forEach((item) => {
  item.addEventListener("click", () => {
    emailText.value = examples[item.dataset.sample];
    emailText.dispatchEvent(new Event("input"));
    emailText.focus();
  });
});

button.addEventListener("click", async () => {
  const text = emailText.value.trim();

  error.textContent = "";

  if (!text) {
    error.textContent = "Please paste an email message before analyzing.";
    return;
  }

  button.disabled = true;
  button.querySelector("span").textContent = "Analyzing...";

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email_text: text,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction could not be completed.");
    }

    showResult(data);
  } catch (err) {
    error.textContent =
      "Backend is not running. Start FastAPI using: uvicorn main:app --reload";
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Analyze email";
  }
});

function showResult(data) {
  const isSpam = data.prediction.toLowerCase() === "spam";

  result.hidden = false;
  result.className = `result-card ${isSpam ? "spam" : "safe"}`;

  document.querySelector("#resultTitle").textContent = isSpam
    ? "Spam detected — Be careful"
    : "No spam detected";

  document.querySelector("#resultDescription").textContent = isSpam
    ? "This email has been classified as spam. Avoid clicking links, opening unknown attachments, or sharing personal details."
    : "This email has been classified as not spam. Still verify unexpected links, attachments, and sender details.";

  result.scrollIntoView({
    behavior: "smooth",
    block: "nearest",
  });
}
