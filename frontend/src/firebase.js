import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signInWithRedirect, signOut } from "firebase/auth";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID,
};

const hasFirebaseConfig = Boolean(
  firebaseConfig.apiKey &&
    firebaseConfig.authDomain &&
    firebaseConfig.projectId &&
    firebaseConfig.storageBucket &&
    firebaseConfig.messagingSenderId &&
    firebaseConfig.appId,
);

let app = null;
let auth = null;
let provider = null;

if (hasFirebaseConfig) {
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
  provider = new GoogleAuthProvider();
  provider.addScope("profile");
  provider.addScope("email");
  provider.setCustomParameters({ prompt: "select_account" });
}

export async function signInWithGoogle() {
  if (!auth || !provider) {
    throw new Error("Firebase Google sign-in is not configured. Add the Firebase env values.");
  }

  try {
    const result = await signInWithPopup(auth, provider);
    return result.user;
  } catch (error) {
    const message = String(error?.message || "");
    const isPopupBlocked =
      error?.code === "auth/popup-blocked" ||
      error?.code === "auth/popup-closed-by-user" ||
      message.includes("Cross-Origin-Opener-Policy") ||
      message.includes("window.closed");

    if (isPopupBlocked) {
      await signInWithRedirect(auth, provider);
      return null;
    }

    if (error && error.code === "auth/configuration-not-found") {
      throw new Error("Firebase Auth is not enabled for this project. Turn on Google sign-in in Firebase Authentication.");
    }

    throw new Error(message || "Could not sign in with Google.");
  }
}

export async function signOutFromGoogle() {
  if (!auth) {
    return;
  }
  await signOut(auth);
}

export function getFirebaseAuth() {
  return auth;
}

export function isFirebaseConfigured() {
  return Boolean(auth && provider);
}
