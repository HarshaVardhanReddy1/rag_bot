import { useCallback, useEffect, useRef, useState } from "react";
import { AuthView } from "./components/AuthView";
import { ChatView } from "./components/ChatView";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const TOKEN_STORAGE_KEY = "project-rag-token";
const USER_STORAGE_KEY = "project-rag-user-email";
const SUPPORTED_DOCUMENT_TYPES = [".pdf", ".txt", ".md"];

const defaultRegisterForm = { name: "", email: "", password: "" };
const defaultLoginForm = { email: "", password: "" };

function buildUrl(path) {
  return `${API_BASE_URL}${path}`;
}

function buildAssetUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return buildUrl(path);
}

function createTempMessage(role, content, extra = {}) {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content,
    created_at: new Date().toISOString(),
    ...extra,
  };
}

function getChatValidation({ prompt, documentFile, selectedChatId }) {
  const errors = {};
  const trimmedPrompt = prompt.trim();

  if (!selectedChatId) {
    errors.form = "Create a chat before sending a message.";
  }

  if (!trimmedPrompt && !documentFile) {
    errors.prompt = "Write a message or attach a document before sending.";
  }

  if (documentFile) {
    const fileName = documentFile.name || "";
    const extension = fileName.slice(fileName.lastIndexOf(".")).toLowerCase();

    if (!SUPPORTED_DOCUMENT_TYPES.includes(extension)) {
      errors.file = `Unsupported file type. Allowed types: ${SUPPORTED_DOCUMENT_TYPES.join(", ")}`;
    }
  }

  return errors;
}

async function apiRequest(path, options = {}) {
  const response = await fetch(buildUrl(path), options);
  let payload = null;

  const responseType = response.headers.get("content-type") || "";
  if (responseType.includes("application/json")) {
    payload = await response.json();
  } else {
    payload = await response.text();
  }

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload?.detail
        ? payload.detail
        : typeof payload === "string" && payload
          ? payload
          : "Something went wrong.";
    throw new Error(message);
  }

  return payload;
}

function formatTime(value) {
  if (!value) return "Just now";

  const normalizedValue =
    typeof value === "string" && !/[zZ]|[+-]\d{2}:\d{2}$/.test(value)
      ? `${value}Z`
      : value;

  const date = new Date(normalizedValue);
  if (Number.isNaN(date.getTime())) return "Just now";

  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Kolkata",
  }).format(date);
}

function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => localStorage.getItem(key) || initialValue);

  useEffect(() => {
    if (value) {
      localStorage.setItem(key, value);
    } else {
      localStorage.removeItem(key);
    }
  }, [key, value]);

  return [value, setValue];
}

function Flash({ flash, onDismiss }) {
  if (!flash.text) return null;

  return (
    <div className={`flash flash-${flash.type || "info"}`} role="alert">
      <span>{flash.text}</span>
      <button type="button" onClick={onDismiss} aria-label="Dismiss message">
        x
      </button>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState("login");
  const [token, setToken] = useLocalStorage(TOKEN_STORAGE_KEY, "");
  const [userEmail, setUserEmail] = useLocalStorage(USER_STORAGE_KEY, "");

  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatTitle, setChatTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [documentFile, setDocumentFile] = useState(null);
  const [uploadInputKey, setUploadInputKey] = useState(0);
  const [flash, setFlash] = useState({ type: "", text: "" });
  const [chatValidation, setChatValidation] = useState({});

  const [isAuthLoading, setIsAuthLoading] = useState(false);
  const [isChatsLoading, setIsChatsLoading] = useState(false);
  const [isMessagesLoading, setIsMessagesLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isCreatingChat, setIsCreatingChat] = useState(false);
  const [isUploadingDocument, setIsUploadingDocument] = useState(false);

  const messagesEndRef = useRef(null);

  const showFlash = useCallback((type, text) => {
    setFlash({ type, text });
  }, []);

  const clearFlash = useCallback(() => {
    setFlash({ type: "", text: "" });
  }, []);

  const loadChats = useCallback(
    async (preferredChatId) => {
      if (!token) return;

      setIsChatsLoading(true);
      try {
        const data = await apiRequest("/chatList", {
          headers: { Authorization: `Bearer ${token}` },
        });

        setChats(data);

        if (!data.length) {
          setSelectedChatId("");
          setMessages([]);
          return;
        }

        if (preferredChatId) {
          setSelectedChatId(preferredChatId);
          return;
        }

        if (!selectedChatId || !data.some((chat) => chat.chat_id === selectedChatId)) {
          setSelectedChatId(data[0].chat_id);
        }
      } catch (error) {
        showFlash("error", error.message);
      } finally {
        setIsChatsLoading(false);
      }
    },
    [selectedChatId, showFlash, token],
  );

  const loadMessages = useCallback(
    async (chatId) => {
      if (!token || !chatId) return;

      setIsMessagesLoading(true);
      try {
        const data = await apiRequest(`/chat/${chatId}/messages`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        setMessages(
          (data.messages || []).map((message, index) => ({
            id: message._id || `${message.role}-${message.created_at || index}-${index}`,
            ...message,
            image_url: buildAssetUrl(message.image_url),
          })),
        );
      } catch (error) {
        showFlash("error", error.message);
      } finally {
        setIsMessagesLoading(false);
      }
    },
    [showFlash, token],
  );

  useEffect(() => {
    if (!token) return;
    loadChats();
  }, [loadChats, token]);

  useEffect(() => {
    if (!token || !selectedChatId) return;
    loadMessages(selectedChatId);
  }, [loadMessages, selectedChatId, token]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleRegister = useCallback(
    async (form, onSuccess) => {
      setIsAuthLoading(true);
      try {
        const payload = await apiRequest("/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
        });

        showFlash("success", payload.message || "Registration complete. You can sign in now.");
        setMode("login");
        onSuccess(defaultRegisterForm);
      } catch (error) {
        showFlash("error", error.message);
      } finally {
        setIsAuthLoading(false);
      }
    },
    [showFlash],
  );

  const handleLogin = useCallback(
    async (form, onSuccess) => {
      setIsAuthLoading(true);
      try {
        const body = new URLSearchParams();
        body.set("username", form.email);
        body.set("password", form.password);

        const payload = await apiRequest("/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body,
        });

        setToken(payload.access_token);
        setUserEmail(form.email);
        clearFlash();
        onSuccess(defaultLoginForm);
      } catch (error) {
        showFlash("error", error.message);
      } finally {
        setIsAuthLoading(false);
      }
    },
    [clearFlash, setToken, setUserEmail, showFlash],
  );

  const handleCreateChat = useCallback(
    async (event) => {
      event.preventDefault();
      setIsCreatingChat(true);

      try {
        const payload = await apiRequest("/newChat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ title: chatTitle.trim() || "New Chat" }),
        });

        setChatTitle("");
        clearFlash();
        await loadChats(payload.chat_id);
      } catch (error) {
        showFlash("error", error.message);
      } finally {
        setIsCreatingChat(false);
      }
    },
    [chatTitle, clearFlash, loadChats, showFlash, token],
  );

  const handleSendMessage = useCallback(
    async (event) => {
      event.preventDefault();

      const validationErrors = getChatValidation({ prompt, documentFile, selectedChatId });
      if (Object.keys(validationErrors).length) {
        setChatValidation(validationErrors);
        showFlash(
          "error",
          validationErrors.form || validationErrors.file || validationErrors.prompt,
        );
        return;
      }

      const promptText = prompt.trim();
      const messageText = promptText || `Uploaded document: ${documentFile.name}`;
      const userMessage = createTempMessage("user", messageText);
      const assistantMessageId = `assistant-${Date.now()}`;

      setIsSending(true);
      setIsUploadingDocument(Boolean(documentFile));
      setMessages((current) => [
        ...current,
        userMessage,
        createTempMessage("assistant", "", { id: assistantMessageId }),
      ]);
      setChatValidation({});
      setPrompt("");
      clearFlash();

      try {
        const formData = new FormData();
        formData.append("chat_id", selectedChatId);
        formData.append("query", promptText);
        if (documentFile) {
          formData.append("file", documentFile);
        }

        const payload = await apiRequest("/chat", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        });

        const assistantText =
          typeof payload === "object" && payload?.answer
            ? payload.answer
            : "No response received.";

        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessageId
              ? { ...message, content: assistantText }
              : message,
          ),
        );

        setDocumentFile(null);
        setUploadInputKey((current) => current + 1);
        await Promise.all([loadMessages(selectedChatId), loadChats(selectedChatId)]);
      } catch (error) {
        setChatValidation((current) => ({
          ...current,
          form: error.message || "Unable to complete the request.",
        }));
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantMessageId
              ? {
                  ...message,
                  content: error.message || "Unable to complete the request.",
                }
              : message,
          ),
        );
        showFlash("error", error.message);
      } finally {
        setIsSending(false);
        setIsUploadingDocument(false);
      }
    },
    [clearFlash, documentFile, loadChats, loadMessages, prompt, selectedChatId, showFlash, token],
  );

  const handleLogout = useCallback(() => {
    setToken("");
    setUserEmail("");
    setChats([]);
    setSelectedChatId("");
    setMessages([]);
    setPrompt("");
    setDocumentFile(null);
    setChatValidation({});
    setUploadInputKey((current) => current + 1);
    showFlash("success", "Session cleared.");
  }, [setToken, setUserEmail, showFlash]);

  const selectedChat = chats.find((chat) => chat.chat_id === selectedChatId) || null;

  return (
    <div className="app-shell">
      <div className="page-wrap">
        <header className="page-header">
          <div>
            <p className="eyebrow">Project Image</p>
            <h1>Simple AI chat with document knowledge upload</h1>
            <p className="page-subtitle">
              Sign in, upload knowledge documents, and chat against your RAG backend.
            </p>
          </div>
          {token ? (
            <button type="button" className="secondary-button header-logout" onClick={handleLogout}>
              Logout
            </button>
          ) : null}
        </header>

        <Flash flash={flash} onDismiss={clearFlash} />

        {!token ? (
          <AuthView
            mode={mode}
            onModeChange={setMode}
            isLoading={isAuthLoading}
            onRegister={handleRegister}
            onLogin={handleLogin}
          />
        ) : (
          <ChatView
            userEmail={userEmail}
            chats={chats}
            selectedChatId={selectedChatId}
            onSelectChat={setSelectedChatId}
            isChatsLoading={isChatsLoading}
            chatTitle={chatTitle}
            onChatTitleChange={setChatTitle}
            onCreateChat={handleCreateChat}
            isCreatingChat={isCreatingChat}
            documentFile={documentFile}
            onDocumentFileChange={(file) => {
              setDocumentFile(file);
              setChatValidation((current) => ({ ...current, file: "", prompt: "", form: "" }));
            }}
            isUploadingDocument={isUploadingDocument}
            uploadInputKey={uploadInputKey}
            selectedChat={selectedChat}
            messages={messages}
            isMessagesLoading={isMessagesLoading}
            formatTime={formatTime}
            prompt={prompt}
            onPromptChange={(value) => {
              setPrompt(value);
              setChatValidation((current) => ({ ...current, prompt: "", form: "" }));
            }}
            onSendMessage={handleSendMessage}
            isSending={isSending}
            messagesEndRef={messagesEndRef}
            validationErrors={chatValidation}
          />
        )}
      </div>
    </div>
  );
}

export default App;
