import { NextRequest } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import { CallbackManager } from "langchain/callbacks";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const { stream, handlers } = LangChainStream();

  const pineconeClient = new Pinecone(
    {
      apiKey: process.env.PINECONE_API_KEY ?? "",
      //environment: "us-east-1-aws",
    }
  );
  //await pineconeClient.init();

  const pineconeIndex = pineconeClient.Index(
    process.env.PINECONE_INDEX_NAME as string
  );

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );
  const vectorStoreRetriever = vectorStore.asRetriever();
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo",
    streaming: true,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });
  const chain = RetrievalQAChain.fromLLM(model, vectorStoreRetriever, {
    //k: 1,
    returnSourceDocuments: true,
  });

  chain.invoke({ query: body.prompt }).catch(console.error);

  return new StreamingTextResponse(stream);
}
