'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import styles from './signup.module.css'; // reuse same styles for simplicity

export default function SignupPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSignup = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch('http://localhost:8080/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const text = await res.text();
        setError(text || 'Signup failed');
        return;
      }

      // Optional: auto-login after signup
      const data = await res.json();
      localStorage.setItem('token', data.token);
      router.push('/dashboard');
    } catch (err) {
      console.error(err);
      setError('Something went wrong');
    }
  };

  return (
    <div className={styles.container}>
      <form onSubmit={handleSignup} className={styles.card}>
        <h2 className={styles.title}>Sign Up</h2>

        <input
          type="text"
          placeholder="Username"
          className={styles.input}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />

        <input
          type="password"
          placeholder="Password"
          className={styles.input}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {error && <p className={styles.error}>{error}</p>}

        <button type="submit" className={styles.button}>
          Sign Up
        </button>
        <p>
  Already registered? <a href="/login">Login</a>
</p>

      </form>
    </div>
  );
}
