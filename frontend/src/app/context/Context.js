"use client";

import { createContext, useState } from "react";

export const Context = createContext();

const ContextProvider = ({ children }) => {
  const [input, setInput] = useState("");
  const [recentPrompt, setRecentPrompt] = useState("");
  const [history, setHistory] = useState([]);
  const [prevPrompts, setPrevPrompts] = useState([]);
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState("");
  const [theme, setTheme] = useState("light");

  const delayPara = (index, nextWord) => {
    setTimeout(() => {
      setResultData((prev) => prev + nextWord);
    }, 50 * index);
  };

  const newChat = () => {
    setLoading(false);
    setShowResult(false);
    setResultData("");
  };

  const onSent = async (prompt) => {
    if (!prompt?.trim()) return;

    try {
      setLoading(true);
      setShowResult(true);
      setInput("");
      setResultData("");
      setRecentPrompt(prompt);

      // 1. Call Spring Boot backend
      const res = await fetch("http://localhost:8080/llm/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("token")}`, // if needed
        },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) throw new Error("LLM server error");

      const data = await res.json();
      const { taskId, response } = data;

      const bulbIcon = `<img src="/bulb_icon.png" alt="Bulb" class="bulb-glow" />`;
      const boldFormatted = response
        .split("**")
        .map((chunk, i) => (i % 2 ? `<b>${chunk}</b>` : chunk))
        .join("");
      const withBreaks = boldFormatted.split("*").join("<br>");
      const finalResponse = withBreaks.split("###").join(bulbIcon);

      // Animate response
      finalResponse.split(" ").forEach((word, i) =>
        delayPara(i, word + " ")
      );

      setPrevPrompts((prev) =>
        prev.includes(prompt) ? prev : [...prev, prompt]
      );

      setHistory((prev) => [
        ...prev,
        { prompt, response: finalResponse, taskId },
      ]);
    } catch (err) {
      console.error("Error in onSent:", err);
      setResultData("An error occurred. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const contextValue = {
    input,
    setInput,
    recentPrompt,
    setRecentPrompt,
    history,
    setHistory,
    prevPrompts,
    onSent,
    showResult,
    loading,
    resultData,
    theme,
    setTheme,
    newChat,
  };

  return (
    <Context.Provider value={contextValue}>{children}</Context.Provider>
  );
};

export default ContextProvider;
