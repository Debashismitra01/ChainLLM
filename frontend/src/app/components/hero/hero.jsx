"use client";
import "./hero.css";
import { useContext, useState, useRef, useEffect } from "react";
import { Context } from "../../context/Context";
import Profile from "../profile/profile";
import ISRO3DMap from "../map/map"; 

export default function Hero() {
  const {
    onSent,
    recentPrompt,
    showResult,
    loading,
    resultData,
    input,
    setInput,
  } = useContext(Context);

  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [previewURLs, setPreviewURLs] = useState([]);
  const [listening, setListening] = useState(false);
  const [voiceText, setVoiceText] = useState("");
  const [showVoiceBox, setShowVoiceBox] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showMenuDropdown, setShowMenuDropdown] = useState(false);
  const [shareUrl, setShareUrl] = useState("");
  const [linkCopied, setLinkCopied] = useState(false);
  const [pasteNotification, setPasteNotification] = useState(false);
  const [show3DMap, setShow3DMap] = useState(false); 

  const recognitionRef = useRef(null);
  const profileRef = useRef(null);
  const shareModalRef = useRef(null);
  const menuDropdownRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const handler = (e) => {
      if (profileRef.current && !profileRef.current.contains(e.target)) {
        setShowProfile(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (shareModalRef.current && !shareModalRef.current.contains(e.target)) {
        setShowShareModal(false);
        setLinkCopied(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (
        menuDropdownRef.current &&
        !menuDropdownRef.current.contains(e.target)
      ) {
        setShowMenuDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Add paste event listener
  useEffect(() => {
    const handlePaste = async (e) => {
      // Only handle paste if we're not in an input field or if we're in our specific input
      const isInOurInput = e.target === inputRef.current;
      const isInTextInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
      
      if (!isInOurInput && isInTextInput) {
        return; // Let other inputs handle their own paste events
      }

      const clipboardData = e.clipboardData || window.clipboardData;
      const items = clipboardData.items;

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        
        // Check if the item is an image
        if (item.type.indexOf('image') === 0) {
          e.preventDefault(); // Prevent default paste behavior
          
          const blob = item.getAsFile();
          if (blob) {
            // Create a File object from the blob
            const timestamp = new Date().getTime();
            const file = new File([blob], `pasted-image-${timestamp}.png`, {
              type: blob.type,
            });
            
            // Add the file to uploaded files
            const previewUrl = URL.createObjectURL(file);
            setUploadedFiles((prev) => [...prev, file]);
            setPreviewURLs((prev) => [...prev, previewUrl]);
            
            // Show notification
            setPasteNotification(true);
            setTimeout(() => setPasteNotification(false), 2000);
            
            // Focus on input if not already focused
            if (inputRef.current) {
              inputRef.current.focus();
            }
          }
          break;
        }
      }
    };

    // Add event listener to document
    document.addEventListener('paste', handlePaste);
    
    // Cleanup
    return () => {
      document.removeEventListener('paste', handlePaste);
    };
  }, []);

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    
    // Validate file types
    const allowedTypes = [
      'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp',
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
      'application/msword', // .doc
      'text/plain'
    ];
    
    const validFiles = files.filter(file => {
      if (!allowedTypes.includes(file.type)) {
        alert(`File type ${file.type} is not supported. Please upload images, PDFs, DOC, DOCX, or text files.`);
        return false;
      }
      return true;
    });
    
    if (validFiles.length === 0) return;
    
    // Create previews for images, show file icons for documents
    const previews = validFiles.map((file) => {
      if (file.type.startsWith("image/")) {
        return URL.createObjectURL(file);
      }
      return null; // For documents, we'll show file name instead
    });
    
    setUploadedFiles((prev) => [...prev, ...validFiles]);
    setPreviewURLs((prev) => [...prev, ...previews]);
    
    // Clear the input to allow uploading the same file again
    e.target.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Check if we have either text input or files
    if (!input.trim() && uploadedFiles.length === 0) {
      alert("Please enter a message or upload files.");
      return;
    }
    
    try {
      // If we have files, create FormData
      if (uploadedFiles.length > 0) {
        const formData = new FormData();
        formData.append("prompt", input.trim() || "Please analyze these files.");
        uploadedFiles.forEach((file) => {
          formData.append("files", file);
        });
        
        // Clear the form
        setInput("");
        setUploadedFiles([]);
        setPreviewURLs([]);
        
        await onSent(formData);
      } else {
        // Text only
        await onSent(input.trim());
      }
    } catch (error) {
      console.error("Error submitting:", error);
      alert("An error occurred while sending your message. Please try again.");
    }
  };

  const handleFileDelete = (index) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviewURLs((prev) => prev.filter((_, i) => i !== index));
  };

  const handleVoiceInput = () => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition not supported in this browser.");
      return;
    }
    if (!recognitionRef.current) {
      const r = new SpeechRecognition();
      r.lang = "en-US";
      r.interimResults = false;
      r.maxAlternatives = 1;
      r.onstart = () => {
        setListening(true);
        setVoiceText("");
        setShowVoiceBox(true);
      };
      r.onend = () => setListening(false);
      r.onresult = (e) => {
        const t = e.results[0][0].transcript;
        setVoiceText(t);
      };
      r.onerror = (e) => {
        console.error("Speech recognition error:", e.error);
        setListening(false);
        setShowVoiceBox(false);
      };
      recognitionRef.current = r;
    }
    recognitionRef.current.start();
  };

  const handleVoiceCancel = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setListening(false);
    setVoiceText("");
    setShowVoiceBox(false);
  };

  const handleVoiceConfirm = async () => {
    if (!voiceText.trim()) {
      handleVoiceCancel();
      return;
    }
    setShowVoiceBox(false);
    setListening(false);
    setInput("");
    await onSent(voiceText.trim());
    setVoiceText("");
  };

  const handleShare = () => {
    const currentUrl = window.location.href;
    setShareUrl(currentUrl);
    setShowShareModal(true);
    setLinkCopied(false);
  };

  const handleCreateLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setLinkCopied(true);
      setTimeout(() => setLinkCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy link:", err);
      // Fallback for older browsers
      const textArea = document.createElement("textarea");
      textArea.value = shareUrl;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setLinkCopied(true);
      setTimeout(() => setLinkCopied(false), 2000);
    }
  };

  const handleNewChat = () => {
    window.location.reload();
    setShowMenuDropdown(false);
  };

  const handleMenuToggle = () => {
    setShowMenuDropdown((prev) => !prev);
  };

  // Helper function to get file icon based on type
  const getFileIcon = (fileType) => {
    if (fileType.includes('pdf')) return '📄';
    if (fileType.includes('word') || fileType.includes('document')) return '📝';
    if (fileType.includes('text')) return '📰';
    return '📁';
  };

  return (
    <div className="main">
      <div className="nav">
        <p>ISRO CHATBOT</p>
        <div className="nav-actions">
          
          <button 
            className="map-toggle-btn" 
            onClick={() => setShow3DMap(!show3DMap)}
            title="Toggle 3D ISRO Map"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            {show3DMap ? "Hide Map" : "Show 3D Map"}
          </button>

          {/* Share Button */}
          <div className="share-wrapper">
            <button className="share-btn" onClick={handleShare}>
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z" />
              </svg>
              Share
            </button>

            {/* Menu Dropdown */}
            <div className="menu-dropdown-wrapper" ref={menuDropdownRef}>
              <button className="menu-dots" onClick={handleMenuToggle}>
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                </svg>
              </button>

              {showMenuDropdown && (
                <div className="menu-dropdown">
                  <button className="menu-item" onClick={handleNewChat}>
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                    >
                      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                    </svg>
                    New Chat
                  </button>
                  <button className="menu-item" onClick={() => setShow3DMap(!show3DMap)}>
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                    >
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                    </svg>
                    {show3DMap ? "Hide 3D Map" : "Show 3D Map"}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Profile Section */}
          <div className="profile-wrapper" ref={profileRef}>
            <img
              src="/user_icon.png"
              alt="User"
              className="avatar"
              onClick={() => setShowProfile((prev) => !prev)}
            />
            {showProfile && <Profile />}
          </div>
        </div>
      </div>

      {/* Paste Notification */}
      {pasteNotification && (
        <div className="paste-notification">
          <div className="paste-notification-content">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
            Image pasted successfully!
          </div>
        </div>
      )}

      {/* 3D Map Integration */}
      {show3DMap && (
        <div className="map-container">
          <div className="map-header">
            <h2>🛰️ ISRO 3D Interactive Map</h2>
            <button 
              className="map-close-btn"
              onClick={() => setShow3DMap(false)}
              aria-label="Close map"
            >
              ✕
            </button>
          </div>
          <ISRO3DMap mapData={{}} />
        </div>
      )}

      {/* Share Modal */}
      {showShareModal && (
        <div className="share-modal-overlay">
          <div className="share-modal" ref={shareModalRef}>
            <div className="share-modal-header">
              <h3>Share public link to chat</h3>
              <button
                className="share-modal-close"
                onClick={() => {
                  setShowShareModal(false);
                  setLinkCopied(false);
                }}
              >
                ✕
              </button>
            </div>
            <div className="share-modal-content">
              <p className="share-description">
                Your name, custom instructions, and any messages you add after
                sharing stay private.
              </p>
              <div className="share-url-container">
                <input
                  type="text"
                  value={shareUrl}
                  readOnly
                  className="share-url-input"
                />
                <button
                  className={`create-link-btn ${linkCopied ? "copied" : ""}`}
                  onClick={handleCreateLink}
                >
                  🔗 {linkCopied ? "Copied!" : "Create link"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="main-container">
        {!showResult ? (
          <>
            <div className="greet">
              <p>
                <span>Hello, Space Explorer.</span>
              </p>
              <p>How can I help you explore ISRO's missions today?</p>
            </div>

            {/* Updated cards with ISRO-specific content */}
            <div className="cards">
              <div
                className="card"
                onClick={() =>
                  setInput(
                    "Tell me about ISRO's recent Mars and Moon missions"
                  )
                }
              >
                <p>Tell me about ISRO's recent Mars and Moon missions</p>
                <img src="/compass_icon.png" alt="Compass icon" />
              </div>
              <div
                className="card"
                onClick={() =>
                  setInput("Show me ISRO's upcoming space missions and projects")
                }
              >
                <p>Show me ISRO's upcoming space missions and projects</p>
                <img src="/bulb_icon.png" alt="Bulb icon" />
              </div>
              <div
                className="card"
                onClick={() =>
                  setInput("Explain how ISRO's satellite technology works")
                }
              >
                <p>Explain how ISRO's satellite technology works</p>
                <img src="/message_icon.png" alt="Message icon" />
              </div>
              <div
                className="card"
                onClick={() => {
                  setInput("Show me the 3D map of ISRO satellites and launch centers");
                  setShow3DMap(true);
                }}
              >
                <p>Show me the 3D map of ISRO satellites and launch centers</p>
                <img src="/code_icon.png" alt="Code icon" />
              </div>
            </div>
          </>
        ) : (
          <div className="result">
            <div className="result-title">
              <img src="/user_icon.png" alt="User icon" />
              <p className="user-prompt">
                <b>{recentPrompt}</b>
              </p>
            </div>

            <div className="result-data">
              <img src="/gemini_icon.png" alt="Gemini icon" />
              {loading ? (
                <div className="loading">
                  <hr />
                  <hr />
                  <hr />
                </div>
              ) : (
                <p
                  className="response-text"
                  dangerouslySetInnerHTML={{ __html: resultData }}
                ></p>
              )}
            </div>
          </div>
        )}

        {showVoiceBox && (
          <div className="voice-box">
            <div className={`voice-visual ${listening ? "anim" : ""}`}>
              <span></span>
              <span></span>
              <span></span>
              <span></span>
              <span></span>
            </div>
            <p className="voice-text">
              {voiceText || "Listening..."}
            </p>
            <button className="voice-cancel" onClick={handleVoiceCancel}>
              ✖
            </button>
            <button className="voice-confirm" onClick={handleVoiceConfirm}>
              ✔
            </button>
          </div>
        )}

        <div className="main-bottom">
          {uploadedFiles.length > 0 && (
            <div className="preview-area">
              {uploadedFiles.map((file, index) => (
                <div className="preview-item" key={index}>
                  <button
                    className="delete-file-btn"
                    onClick={() => handleFileDelete(index)}
                    aria-label="Delete file"
                  >
                    ✕
                  </button>
                  {previewURLs[index] ? (
                    <img
                      src={previewURLs[index]}
                      alt={`Preview of ${file.name}`}
                      className="preview-image"
                    />
                  ) : (
                    <div className="file-preview">
                      <div className="file-icon">{getFileIcon(file.type)}</div>
                      <span className="file-name">{file.name}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          <form onSubmit={handleSubmit} className="search-box">
            <input
              ref={inputRef}
              type="text"
              onChange={(e) => setInput(e.target.value)}
              value={input}
              placeholder={uploadedFiles.length > 0 ? "Ask about your files..." : "Ask about ISRO missions, satellites, or paste an image..."}
              disabled={loading}
            />
            <div>
              <input
                id="fileUpload"
                type="file"
                multiple
                accept="image/*,.pdf,.doc,.docx,.txt"
                style={{ display: "none" }}
                onChange={handleFileUpload}
              />
              <img
                src="/gallery_icon.png"
                alt="Gallery icon"
                style={{ cursor: "pointer" }}
                onClick={() => document.getElementById("fileUpload").click()}
              />
              <div
                className="mic-wrapper"
                onClick={handleVoiceInput}
                style={{ cursor: "pointer" }}
              >
                <img src="/mic_icon.png" alt="Mic icon" />
              </div>
              <button type="submit" disabled={loading}>
                <img id="send" src="/send_icon.png" alt="Send icon" />
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Additional CSS for the 3D Map Integration */}
      <style jsx>{`
        .map-toggle-btn {
          background: linear-gradient(135deg, #ff6b35, #f7931e);
          border: none;
          color: white;
          padding: 8px 16px;
          border-radius: 20px;
          cursor: pointer;
          font-size: 12px;
          font-weight: 500;
          display: flex;
          align-items: center;
          gap: 6px;
          transition: all 0.3s ease;
          margin-right: 10px;
        }

        .map-toggle-btn:hover {
          background: linear-gradient(135deg, #e55a2b, #e8831a);
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
        }

        .map-container {
          position: fixed;
          top: 80px;
          left: 20px;
          right: 20px;
          bottom: 20px;
          background: rgba(0, 0, 0, 0.95);
          border-radius: 16px;
          z-index: 1000;
          overflow: hidden;
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .map-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          background: linear-gradient(135deg, #1a1a2e, #16213e);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .map-header h2 {
          color: #ff6b35;
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }

        .map-close-btn {
          background: rgba(255, 255, 255, 0.1);
          border: none;
          color: white;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          cursor: pointer;
          font-size: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
        }

        .map-close-btn:hover {
          background: rgba(255, 107, 53, 0.8);
          transform: scale(1.1);
        }

        @media (max-width: 768px) {
          .map-container {
            top: 70px;
            left: 10px;
            right: 10px;
            bottom: 10px;
          }
          
          .map-toggle-btn {
            padding: 6px 12px;
            font-size: 11px;
          }
          
          .map-header {
            padding: 15px;
          }
          
          .map-header h2 {
            font-size: 16px;
          }
        }

        .nav-actions {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .menu-item {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 10px 16px;
          background: none;
          border: none;
          color: #333;
          cursor: pointer;
          font-size: 14px;
          transition: background-color 0.2s ease;
        }

        .menu-item:hover {
          background-color: #f5f5f5;
        }

        .menu-item svg {
          width: 16px;
          height: 16px;
        }
      `}</style>
    </div>
  );
}