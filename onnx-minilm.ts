// @ts-ignore
import { InferenceSession, TypedTensor } from "onnxruntime-node";
import { BertTokenizer } from "@xenova/transformers";
import tokenizerJson from "./models/all-MiniLM-L6-v2/tokenizer.json" assert { type: "json" };
import tokenizerConfigJson from "./models/all-MiniLM-L6-v2/tokenizer_config.json" assert { type: "json" };

function cosineSimilarity(vecA: number[], vecB: number[]) {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function main() {
  // Define the query and sentences
  const query = "I work at Kin";
  const sentences = [
    "thomas works at Kin",
    "kenji works at Kin",
    "Christopher works at-> Kin",
    "Me co-founded Kin",
    "Andrei works at Kin",
    "Me is cto of Kin",
    "Volodymyr works at Kin",
    "Me reached at kin feature freeze",
    "Me vents to Kin",
    "0.1.1 version of Kin",
    "Kin offers help Me",
    "0.1.4 version of Kin",
    "Kin requires responsibility",
    "Me looks for in candidates align with Kin's values",
    "Kin values transparency and trustworthiness",
    "Me works at Kin",
    "I work at Kin",
    "Me works at Kin",
  ];

  // Load the ONNX model
  const session = await InferenceSession.create(
    "./models/all-MiniLM-L6-v2/onnx/model.onnx"
  );

  // const tokenizer = new PreTrainedTokenizer(tokenizerJson, tokenizerConfigJson);
  const tokenizer = new BertTokenizer(tokenizerJson, tokenizerConfigJson);

  // Tokenize and prepare input
  // const prepareInput = (text: string) => {
  //   const encodedText = tokenizer._call(text);

  //   // @ts-ignore
  //   const maxLength = encodedText.input_ids.data.length;

  //   // @ts-ignore
  //   const ids = Object.values(encodedText.input_ids.data).map((bigIntValue) =>
  //     BigInt(bigIntValue as number)
  //   );

  //   // @ts-ignore
  //   const mask = Object.values(encodedText.attention_mask.data).map(
  //     (bigIntValue) => BigInt(bigIntValue as number)
  //   );

  //   // @ts-ignore
  //   const token_type_ids = encodedText.attention_mask.clone();
  //   token_type_ids.data.fill(0n);

  //   const tokenType = Object.values(token_type_ids.data).map((bigIntValue) =>
  //     BigInt(bigIntValue as number)
  //   );

  //   // Padding to ensure all arrays are of equal length
  //   while (ids.length < maxLength) {
  //     ids.push(0n);
  //     mask.push(0n);
  //     tokenType.push(0n);
  //   }

  //   const batchInputIds = new Tensor("int64", ids.flat(), [1, maxLength]);

  //   const batchAttentionMask = new Tensor("int64", mask.flat(), [1, maxLength]);

  //   const batchTokenTypeId = new Tensor("int64", tokenType.flat(), [
  //     1,
  //     maxLength,
  //   ]);

  //   const inputs = {
  //     input_ids: batchInputIds,
  //     attention_mask: batchAttentionMask,
  //     token_type_ids: batchTokenTypeId,
  //   };

  //   return inputs;
  // };

  // Tokenize and prepare input
  const prepareInput = (text: string) => {
    const model_inputs = tokenizer._call(text, {
      padding: true,
      truncation: true,
    });

    // Delete all the other input preperation

    return model_inputs;
  };

  const queryInput = prepareInput(query);
  const sentenceInputs = sentences.map(prepareInput);

  // Function to extract embeddings
  const extractEmbedding = async (input: {
    input_ids: TypedTensor<"int64">;
    attention_mask: TypedTensor<"int64">;
    token_type_ids?: TypedTensor<"int64">;
  }) => {
    let output = await session.run(input);
    // output = mean_pooling(output.last_hidden_state, input.attention_mask);
    // output = output.normalize(2, -1);

    return Array.from(output.last_hidden_state.data);
    // return Array.from(output.data);
  };

  // Compute embeddings
  const queryEmbedding = await extractEmbedding(queryInput);

  const sentenceEmbeddings = await Promise.all(
    sentenceInputs.map(extractEmbedding)
  );

  // Compute cosine similarities
  const cosineScores = sentenceEmbeddings.map((sentenceEmbedding) =>
    // @ts-ignore
    cosineSimilarity(queryEmbedding, sentenceEmbedding)
  );

  // Combine sentences with their scores and sort by similarity
  const sentenceScorePairs = sentences.map((sentence, index) => {
    console.log(sentenceInputs[index].input_ids);
    return {
      sentence: sentence,
      score: cosineScores[index],
      embeddings: sentenceEmbeddings[index],
    };
  });

  const sortedSentences = sentenceScorePairs.sort((a, b) => b.score - a.score);

  // Output the sorted sentences with their similarity scores
  sortedSentences.forEach((pair) => {
    console.log(`${pair.sentence}, ${pair.score}`);
  });

  console.log("sortedSentences", JSON.stringify(sortedSentences));
}

main().catch(console.error);
