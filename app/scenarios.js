// Practice scenarios for Presence. Each gives on-screen prompts to work through while it reads you.
export const SCENARIOS = [
  {
    id: "free", label: "Free practice", icon: "ph-microphone-stage",
    blurb: "Just talk. Presence reads you the whole time.",
    prompts: ["Speak about anything for a minute.", "Tell a short story you know well."],
  },
  {
    id: "interview", label: "Job interview", icon: "ph-briefcase",
    blurb: "Answer common interview questions with composure.",
    prompts: [
      "Tell me about yourself.",
      "Why do you want this role?",
      "Describe a time you handled conflict.",
      "What is your biggest weakness?",
      "Where do you see yourself in five years?",
      "Why should we hire you?",
    ],
  },
  {
    id: "pitch", label: "Pitch / present", icon: "ph-presentation-chart",
    blurb: "Deliver a crisp, confident pitch.",
    prompts: [
      "What is the one-line version of your idea?",
      "What problem are you solving, and for whom?",
      "Why now, and why you?",
      "What is the ask?",
    ],
  },
  {
    id: "hard", label: "Hard conversation", icon: "ph-chats-circle",
    blurb: "Stay calm and open through a difficult talk.",
    prompts: [
      "Open the conversation kindly but directly.",
      "State the issue without blame.",
      "Acknowledge the other side's view.",
      "Propose a way forward.",
    ],
  },
];
export const scenarioById = (id) => SCENARIOS.find((s) => s.id === id) || SCENARIOS[0];
