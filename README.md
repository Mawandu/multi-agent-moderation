# Multi-Agent Moderation System 🛡️

A decentralized content moderation system powered by multi-agent LLM orchestration using LangChain and Groq. This Streamlit application simulates a committee of specialized AI agents that collaborate to analyze content and reach a consensus decision.

## 🌟 Features

*   **Multi-Agent Architecture**:
    *   **Text Analysis Agent**: Detects hate speech and harmful language.
    *   **Image Recognition Agent**: Simulates visual analysis for nudity, violence, etc.
    *   **Cultural Context Agent**: Flags culturally sensitive phrases based on user jurisdiction.
    *   **Legal Compliance Agent**: Checks against regional laws (GDPR, DSA, Section 230).
*   **Arbitration Engine**: Resolves conflicts between agents using weighted voting and confidence levels.
*   **Interactive UI**: Built with Streamlit for real-time testing and visualization.
*   **Monitoring System**: Tracks metrics like false positive rates and hard limit hits.

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   A [Groq API Key](https://console.groq.com/)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Mawandu/multi-agent-moderation.git
    cd multi-agent-moderation
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Configure environment variables:
    *   Rename `.env.example` to `.env`.
    *   Add your Groq API key: `GROQ_API_KEY=your_key_here`.
    *   *Alternatively, you can enter the key in the app's sidebar.*

### Running the App

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **AI/LLM**: LangChain, Groq API (Llama3, Mixtral)
*   **Language**: Python

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
