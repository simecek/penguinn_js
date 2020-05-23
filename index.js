
async function makePrediction() {
  
  var prob = document.getElementById('prob2');
  var seq = document.getElementById('text1');
  
  console.log("model loading..");

  // clear the model variable
  var model = undefined;
  
  model = await tf.loadLayersModel("https://raw.githubusercontent.com/simecek/penguinn_js/master/assets/model.json");
  
  console.log("model loaded...");

  const s = seq.value.replace(/\r?\n|\r/g,'').replace(/\s/g, '');  
  const t = s.replace(/A/g,'0').replace(/T/g,'1').replace(/U/g,'1').replace(/C/g,'2').replace(/G/g,'3').replace(/N/g,'9')
  const y = tf.oneHot(tf.tensor1d(t.split(''),'int32'),4);
  const z = y.reshape([1,200,4]);
  
  const result = model.predict(z).asScalar().dataSync();

  console.log("prediction done...");
  
  prob.innerHTML = "<br><br>Probability of G4 complex:   " + result;
}

// Example: 
// GAGACACCACTACAGTTAGCAGTGAGTGTAAAATAATGAGTGTCAGAAACTTATATTGGGTGATTTCATTTTTAAAAGTAACCAAAGTGAAAAATGAAGCCTTGCGTTTTTGCTTAAATGATTTACAAAAAATATTTGATGTCCATCCTGGGATAGGGAATTCCTCCCCCATAACTTTGAAAGTGCAGTTGCTTCATTCC