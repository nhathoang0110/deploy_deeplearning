let model;
let IMAGE_WIDTH = 300;



async function loadModel() {
	console.log("model loading model kdfah ...");
	loader = document.getElementById("progress-box");
	load_button = document.getElementById("load-button");
	loader.style.display = "block";
	modelName = "mobilenet";
	model = undefined;
	
	//model = await tf.loadLayersModel('models1/mobile/model.json');
	model = await tf.loadLayersModel('models1/vgg_v2/model.json');

	


	if (typeof model !== "undefined") {
		loader.style.display = "none";
		load_button.disabled = true;		
		load_button.innerHTML = "Loaded Model";
		console.log("model loaded..");
	}
};

function loadImageLocal() {
	console.log("Click into selected file image");
  	document.getElementById("select-file-box").style.display = "table-cell";
  	document.getElementById("predict-box").style.display = "table-cell";
  	document.getElementById("prediction").innerHTML = "Click predict to find my label!";
    renderImage(this.files);
};




function renderImage(file) {
  var reader = new FileReader();
  reader.onload = function(event) {
    let output = document.getElementById('test-image');
  	output.src = reader.result;
  	output.width = IMAGE_WIDTH;
  }
  
  if(event.target.files[0]){
	reader.readAsDataURL(event.target.files[0]);
  }
}

//const webcamElement = document.getElementById('webcam');
async function predictImage(){

	console.log("model loading model kdfah ...");
	loader = document.getElementById("progress-box");
	load_button = document.getElementById("load-button");
	loader.style.display = "block";
	modelName = "mobilenet";
	model = undefined;
	
	//model = await tf.loadLayersModel('models1/mobile/model.json');
	model = await tf.loadLayersModel('models1/vgg_v2/model.json');

	


	// if (typeof model !== "undefined") {
	// 	loader.style.display = "none";
	// 	//load_button.disabled = true;		
	// 	load_button.innerHTML = "Loaded Model";
	// 	console.log("model loaded..");
	// }

	console.log("Click predict button");
	if (model == undefined) {
		alert("Please load the model first..")
	}
	// if (document.getElementById("predict-box").style.display == "none") {
	// 	alert("Please load an image using 'Upload Image' button..")
	// }
	

	let webcamElement = document.getElementById('webcam');



	
	let video = document.getElementById("video");
	let scale =  1;
	const canvas = document.createElement("canvas");
    //canvas.width = await video.clientWidth * scale;
    //canvas.height = await video.clientHeight * scale;
    //await canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
	
	//const image = new Image()
	//image.src = canvas.toDataURL();
	//image1= await tf.browser.fromPixels(canvas)
	
	


	//let webcam = await tf.data.webcam(webcamElement);
	
	
	var arr=[0,0,0,0,0,0,0,0,0,0]
	let i=0
	//arr[0]='0'
	var dem1=0
	var dem2=0
	var cnt = 0;
	while(true){

		var t0 = performance.now()
		//const img = await  webcam.capture();

		canvas.width = video.clientWidth * scale;
		canvas.height = video.clientHeight * scale;
		await canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);


		// canvas.width = webcamElement.clientWidth * scale;
		// canvas.height = webcamElement.clientHeight * scale;
		// await canvas.getContext('2d').drawImage(webcamElement, 0, 0, canvas.width, canvas.height);
		
		// const image=new Image()
		// //image.src=canvas.toDataURL()
		// image.src=canvas.toDataURL()
		// document.getElementById("imageshow").innerHTML= '<img src="' + image.src + '" width=224px" height="224px" /> ';

		img= await tf.browser.fromPixels(canvas)
	

		//let tensor= await img.resizeNearestNeighbor([224, 224]).toFloat().expandDims();
		let tensor= await img.reverse(-1).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
		//images
		//let tensor = await tf.browser.fromPixels(image).toFloat().expandDims();
		
		let predictions = await model.predict(tensor).data();
		var t1 = performance.now()
		console.log(t1-t0)
		predictions[6]=predictions[6]-0.3
		
		let results = Array.from(predictions)
			.map(function (p, i) {
			return {
				probability: p,
				className: IMAGENET_CLASSES[i]
			}; 
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 5);

		//arr.push(results[0].className);
		// console.log(arr[i]);
		// if(arr[i]==arr[i-1]) dem1++;
		// if(arr[i]!=arr[i-1]) dem2++;

		// if(int(result[0].className) == 0) arr[]
		cnt += 1;
		str = 'mtt' + results[0].className;
		for(j = 0; j < 10; j++)
			if (j != results[0].className) {
				str2 = 'mtt' + j;
				document.getElementById(str2).setAttribute('style', 'color: white')
			}
		document.getElementById(str).setAttribute('style', 'color: red')
		arr[results[0].className]+=1
		i=i+1;
		console.log(str);
		console.log(i)
		if(i==10){
			arr[8]=arr[8]
			arr[6]=arr[6]
			max=0
			console.log(max)
			for(k=0;k<10;k++){
				if(arr[k]>max){
					max=arr[k];
				}			
			}
			for(k=0;k<10;k++){
			if(arr[k]==max && k!=0){
				document.getElementById("canhbao").innerHTML = "WARNING: Đang lái xe không an toàn";
			}
			if(arr[k]==max && k==0){
				document.getElementById("canhbao").innerHTML = "Binh thuong";
			}
		
		}
			

			for(k=0;k<10;k++){
				arr[k]=0;
			}
			i=0;
		}
 


		// console.log(dem1)
		// console.log(dem2)
		// if(results[0].className != '0' && dem1 >= 10){
		// 	document.getElementById("canhbao").innerHTML = "canh bao <br><b>" + results[0].className + "</b>";
		// }else{
		// 	document.getElementById("canhbao").innerHTML = "canh bao <br><b>" + "Binh thuong" + "</b>";
		// }


		
		img.dispose();
		tf.nextFrame();
	}
	
}

function preprocessImage(image, modelName) {
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224])
		.toFloat();


	if (modelName === undefined) {
		return tensor.expandDims();
	} else if (modelName === "mobilenet") {
		let offset = tf.scalar(127.5);
		return tensor.sub(offset)
			.div(offset)
			.expandDims();
	} else {
		alert("Unknown model name..")
	}
}

