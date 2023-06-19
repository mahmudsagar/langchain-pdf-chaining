import dotenv from "dotenv"
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";
const model = new ChatOpenAI({openAIApiKey: process.env.OPENAI_API_KEY, temperature: 0 });
dotenv.config();
// const llmA = new OpenAI({});
const chainA = loadQAStuffChain(model);
const loader = new PDFLoader("./src/express-handbook.pdf");

const docs = await loader.load();

const resA = await chainA.call({
  input_documents: docs,
  question: "write hello world with express",
});
console.log({ resA });
