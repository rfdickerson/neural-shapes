import "./styles.css";
import { App } from "./app";

const root = document.getElementById("app");
if (!root) {
  throw new Error("Missing #app root element.");
}

const app = new App(root);
app.start();
