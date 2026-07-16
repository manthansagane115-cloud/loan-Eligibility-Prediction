document.addEventListener("DOMContentLoaded", () => {
  const steps = document.querySelectorAll(".form-step");
  let currentStep = 0;

  const showStep = (index) => {
    steps.forEach((step, i) => {
      step.classList.toggle("hidden", i !== index);
    });
  };

  document.querySelectorAll(".next-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (currentStep < steps.length - 1) {
        currentStep++;
        showStep(currentStep);
      }
    });
  });

  document.querySelectorAll(".prev-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      if (currentStep > 0) {
        currentStep--;
        showStep(currentStep);
      }
    });
  });

  showStep(currentStep);
});
