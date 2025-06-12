import express, { type Response } from "express";
const app = express();

app.use(express.json());

app.get("/", (_, res: Response) => {
    res.send("Hello World");
});

app.listen(3000, () => {
    console.log("Server is running on port 3000");
});

export default app;