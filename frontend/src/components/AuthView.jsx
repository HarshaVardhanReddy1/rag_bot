import { useState } from "react";

const defaultRegisterForm = { name: "", email: "", password: "" };
const defaultLoginForm = { email: "", password: "" };

export function AuthView({ mode, onModeChange, isLoading, onRegister, onLogin }) {
  const [registerForm, setRegisterForm] = useState(defaultRegisterForm);
  const [loginForm, setLoginForm] = useState(defaultLoginForm);

  return (
    <main className="auth-layout">
      <section className="card intro-card">
        <p className="eyebrow">Overview</p>
        <h2>Everything important stays visible.</h2>
        <p>
          This frontend keeps the layout simple: auth on one side, product context on the other,
          with no oversized hero sections pushing content below the fold.
        </p>
        <div className="info-list">
          <div>
            <strong>Auth</strong>
            <span>Register and login with your FastAPI backend.</span>
          </div>
          <div>
            <strong>Chats</strong>
            <span>Create threads and reopen previous conversations.</span>
          </div>
          <div>
            <strong>Images</strong>
            <span>Attach a file and send it with your prompt.</span>
          </div>
        </div>
      </section>

      <section className="card auth-card">
        <div className="tabs">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => onModeChange("login")}
          >
            Login
          </button>
          <button
            type="button"
            className={mode === "register" ? "active" : ""}
            onClick={() => onModeChange("register")}
          >
            Register
          </button>
        </div>

        {mode === "register" ? (
          <form
            className="form-grid"
            onSubmit={(event) => {
              event.preventDefault();
              onRegister(registerForm, () => {
                setRegisterForm(defaultRegisterForm);
                setLoginForm((current) => ({ ...current, email: registerForm.email }));
              });
            }}
          >
            <label>
              <span>Name</span>
              <input
                type="text"
                value={registerForm.name}
                onChange={(event) =>
                  setRegisterForm((current) => ({ ...current, name: event.target.value }))
                }
                placeholder="Harsha"
                required
                disabled={isLoading}
              />
            </label>
            <label>
              <span>Email</span>
              <input
                type="email"
                value={registerForm.email}
                onChange={(event) =>
                  setRegisterForm((current) => ({ ...current, email: event.target.value }))
                }
                placeholder="you@example.com"
                required
                disabled={isLoading}
              />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                value={registerForm.password}
                onChange={(event) =>
                  setRegisterForm((current) => ({ ...current, password: event.target.value }))
                }
                placeholder="At least 8 characters"
                required
                disabled={isLoading}
              />
            </label>
            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "Creating..." : "Create account"}
            </button>
          </form>
        ) : (
          <form
            className="form-grid"
            onSubmit={(event) => {
              event.preventDefault();
              onLogin(loginForm, () => setLoginForm(defaultLoginForm));
            }}
          >
            <label>
              <span>Email</span>
              <input
                type="email"
                value={loginForm.email}
                onChange={(event) =>
                  setLoginForm((current) => ({ ...current, email: event.target.value }))
                }
                placeholder="you@example.com"
                required
                disabled={isLoading}
              />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                value={loginForm.password}
                onChange={(event) =>
                  setLoginForm((current) => ({ ...current, password: event.target.value }))
                }
                placeholder="Your password"
                required
                disabled={isLoading}
              />
            </label>
            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "Signing in..." : "Enter workspace"}
            </button>
          </form>
        )}
      </section>
    </main>
  );
}
