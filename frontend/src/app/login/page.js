'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import styles from './login.module.css';

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch('http://localhost:8080/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        setError('Invalid credentials');
        return;
      }

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
      <form onSubmit={handleLogin} className={styles.card}>
        <h2 className={styles.title}>Login</h2>

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
          Log In
        </button>
         <p>Don't have an account? <a href='/signup'>Create Account</a></p>
      </form>
    </div>
  );
}
