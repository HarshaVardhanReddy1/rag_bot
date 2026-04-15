export function ChatView({
  userEmail,
  chats,
  selectedChatId,
  onSelectChat,
  isChatsLoading,
  chatTitle,
  onChatTitleChange,
  onCreateChat,
  isCreatingChat,
  documentFile,
  onDocumentFileChange,
  isUploadingDocument,
  uploadInputKey,
  selectedChat,
  messages,
  isMessagesLoading,
  formatTime,
  prompt,
  onPromptChange,
  onSendMessage,
  isSending,
  messagesEndRef,
  validationErrors,
}) {
  return (
    <main className="chat-layout">
      <aside className="card sidebar-card">
        <div className="sidebar-top">
          <div>
            <p className="eyebrow">Account</p>
            <h2>{userEmail || "User"}</h2>
          </div>
        </div>

        <form className="form-grid compact-form" onSubmit={onCreateChat}>
          <label>
            <span>New chat</span>
            <input
              type="text"
              value={chatTitle}
              onChange={(event) => onChatTitleChange(event.target.value)}
              placeholder="Product brainstorm"
              disabled={isCreatingChat}
            />
          </label>
          <button className="primary-button" type="submit" disabled={isCreatingChat}>
            {isCreatingChat ? "Creating..." : "Create chat"}
          </button>
        </form>

        <div className="chat-list">
          <div className="list-head">
            <span>Chats</span>
            <strong>{isChatsLoading ? "..." : chats.length}</strong>
          </div>

          {isChatsLoading ? (
            <div className="empty-box">Loading chats...</div>
          ) : chats.length ? (
            chats.map((chat) => (
              <button
                key={chat.chat_id}
                type="button"
                className={`chat-item ${selectedChatId === chat.chat_id ? "active" : ""}`}
                onClick={() => onSelectChat(chat.chat_id)}
              >
                <strong>{chat.title}</strong>
                {chat.updated_at ? <time>{formatTime(chat.updated_at)}</time> : null}
              </button>
            ))
          ) : (
            <div className="empty-box">Create a chat to begin.</div>
          )}
        </div>
      </aside>

      <section className="card conversation-card">
        <div className="conversation-top">
          <div>
            <p className="eyebrow">Conversation</p>
            <h2>{selectedChat?.title || "Select or create a chat"}</h2>
          </div>
        </div>

        <div className="messages-box">
          {isMessagesLoading ? (
            <div className="empty-box large">Loading thread...</div>
          ) : messages.length ? (
            <>
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`message-row ${message.role === "assistant" ? "assistant" : "user"}`}
                >
                  <div className="message-label">
                    <strong>{message.role === "assistant" ? "Assistant" : "You"}</strong>
                    <span>{formatTime(message.created_at)}</span>
                  </div>
                  <div className="message-bubble">
                    {message.image_url ? (
                      <img
                        className="message-image"
                        src={message.image_url}
                        alt="Uploaded by user"
                      />
                    ) : null}
                    <p>{message.content}</p>
                  </div>
                </article>
              ))}
              <div ref={messagesEndRef} />
            </>
          ) : (
            <div className="empty-box large">Send the first prompt to start this conversation.</div>
          )}
        </div>

        <form className="composer-box" onSubmit={onSendMessage}>
          {validationErrors?.form ? <p className="field-error">{validationErrors.form}</p> : null}

          <label>
            <span>Prompt</span>
            <textarea
              className={validationErrors?.prompt ? "input-error" : ""}
              value={prompt}
              onChange={(event) => onPromptChange(event.target.value)}
              placeholder="Ask something about your documents..."
              rows={4}
              disabled={isSending}
            />
            {validationErrors?.prompt ? (
              <small className="field-error">{validationErrors.prompt}</small>
            ) : null}
          </label>

          <label className="file-field">
            <span>Document attachment (optional)</span>
            <input
              className={validationErrors?.file ? "input-error" : ""}
              key={uploadInputKey}
              type="file"
              accept=".pdf,.txt,.md"
              onChange={(event) => onDocumentFileChange(event.target.files?.[0] || null)}
              disabled={isSending || isUploadingDocument}
            />
            <strong>{documentFile ? documentFile.name : "Accepted: PDF, TXT, MD"}</strong>
            {validationErrors?.file ? (
              <small className="field-error">{validationErrors.file}</small>
            ) : null}
          </label>

          <div className="composer-actions">
            <button
              className="primary-button send-button"
              type="submit"
              disabled={isSending || isUploadingDocument}
            >
              {isUploadingDocument ? "Uploading..." : isSending ? "Sending..." : "Send"}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
