import { NextResponse } from 'next/server';

const protectedPaths = ['/', '/settings', '/profile'];

export function middleware(request) {
  const token = request.cookies.get('token')?.value;
  const { pathname } = request.nextUrl;

  if (protectedPaths.some(path => pathname.startsWith(path))) {
    if (!token) {
      return NextResponse.redirect(new URL('/login', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/', '/settings/:path*', '/profile/:path*'],
};
