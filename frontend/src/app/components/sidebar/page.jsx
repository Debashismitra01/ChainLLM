"use client";
import { useState, useContext } from "react";
import "./sidebar.css";
import { Context } from "../../context/Context"; // Adjust if needed

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [showActivity, setShowActivity] = useState(false);

  const {
    onSent,
    prevPrompts,
    history,
    setRecentPrompt,
    newChat,
  } = useContext(Context);

  const toggleSidebar = (e) => {
    if (e.currentTarget.className.includes("sidebar-icon")) {
      setIsOpen(!isOpen);
    }
  };

  const handlePromptClick = (prompt) => {
    setRecentPrompt(prompt);
    onSent(prompt);
  };

  const handleActivityClick = () => {
    setShowActivity(true);
  };

  const handleRecentClick = () => {
    setShowActivity(false);
  };

  return (
    <div className={`sidebar-container ${isOpen ? "open" : "closed"}`}>
      <div className="top">
        <div className="sidebar-header">
          <img
            src="/sidebar_icon.png"
            className="sidebar-icon"
            alt="Menu"
            onClick={toggleSidebar}
          />
        </div>

        <div onClick={() => newChat()} className="new-chat">
          <div className="plus-container">
            <img className="plus" src="/plus_icon.png" alt="New chat" />
          </div>
          <p>New Chat</p>
        </div>

        <div className="recent">
          <p
            className="recent-title"
            onClick={handleRecentClick}
            style={{ cursor: "pointer" }}
          >
            {showActivity ? "Back to Recent" : "Recent"}
          </p>

          {!showActivity &&
            prevPrompts.map((item, index) => (
              <div
                className="recent-entry"
                key={index}
                onClick={() => handlePromptClick(item)}
              >
                <img className="message" src="/message_icon.png" alt="Message" />
                <p>{item.slice(0, 18)}...</p>
              </div>
            ))}

          {showActivity &&
            history.map((entry, index) => (
              <div
                className="recent-entry"
                key={index}
                onClick={() => handlePromptClick(entry.prompt)}
              >
                <img className="message" src="/history_icon.png" alt="History" />
                <p>{entry.prompt.slice(0, 18)}...</p>
              </div>
            ))}
        </div>
      </div>

      <div className="bottom">
      </div>
    </div>
  );
}
