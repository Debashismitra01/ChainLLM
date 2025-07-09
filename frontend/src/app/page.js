"use client"

import "./page.css";
import Sidebar from "./components/sidebar/page";
import Hero from "./components/hero/hero";
import ContextProvider from "./context/Context";
import useAuthRedirect from "@/lib/useAuthRedirect";

export default function Home() {
    useAuthRedirect();
    return (
        <ContextProvider>
            <div className="app-container">
                <Sidebar />
                <div className="main-content">
                    <Hero />
                </div>
            </div>
        </ContextProvider>
    );
}