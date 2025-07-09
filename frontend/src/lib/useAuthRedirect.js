'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function useAuthRedirect() {
  const router = useRouter();
  const [isChecked, setIsChecked] = useState(false);

  useEffect(() => {
    // Run this only on client after DOM is ready
    const checkAuth = () => {
      try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
        if (!token) {
          router.push('/login');
        } else {
          console.log('token exists');
        }
      } catch (error) {
        console.error('Error accessing localStorage:', error);
        router.push('/login');
      } finally {
        setIsChecked(true);
      }
    };

    checkAuth();
  }, [router]);

  return isChecked;
}
