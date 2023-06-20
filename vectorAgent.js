import dotenv from "dotenv"
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import {
  VectorStoreToolkit,
  createVectorStoreAgent
} from "langchain/agents";

dotenv.config();
console.log(process.env.OPENAI_API_KEY);
const model = new ChatOpenAI({openAIApiKey: process.env.OPENAI_API_KEY, temperature: 0 });
// const model = new OpenAI({});
// const chainA = loadQAStuffChain(model);
const loader = new PDFLoader("./src/express-handbook.pdf");

const docs = await loader.loadAndSplit();

/* Split the text into chunks */
// const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
// const docs = await textSplitter.createDocuments([text]);
/* Create the vectorstore */
const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

/* Create the agent */
const vectorStoreInfo = {
  name: "express js docs",
  description: "this a single  page documentation of express js that has installation and hello world example",
  vectorStore,
};

const toolkit = new VectorStoreToolkit(vectorStoreInfo, model);
const agent = createVectorStoreAgent(model, toolkit);
const input =
  "how to install express js?";
console.log(`Executing: ${input}`);

const result = await agent.call({ input });
console.log(`Got output ${result.output}`);
console.log(
  `Got intermediate steps ${JSON.stringify(result.intermediateSteps, null, 2)}`
);