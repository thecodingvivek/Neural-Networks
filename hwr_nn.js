function formatGrayscaleData(data, width) {
    let result = '';
    for (let i = 0; i < data.length; i++) {
        result += data[i].toString().padStart(3, ' ') + ' ';
        if ((i + 1) % width === 0) {
            result += '\n';
        }
    }
    return result;
}
document.addEventListener('DOMContentLoaded',()=>{
    var o=false;
    var largeCanvas=document.getElementById("Lcanva");
    var smallCanvas=document.getElementById("Scanva");
    var smallCtx=smallCanvas.getContext('2d');
    var progress=document.getElementsByClassName('prediction_rate')[0];


    smallCtx.fillStyle = "black";
    smallCtx.fillRect(0, 0, largeCanvas.width, largeCanvas.height);

    var largeCtx=largeCanvas.getContext('2d');
    let isDrawing = false;

    largeCanvas.addEventListener('mousedown', () => {
        isDrawing = true;
    });

    largeCanvas.addEventListener('mouseup', () => {
        isDrawing = false;
    });

    largeCanvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
            const rect = largeCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            largeCtx.fillStyle = "white";
            largeCtx.beginPath();
            largeCtx.arc(x, y, 10, 0, Math.PI * 2); // Increased brush size
            largeCtx.fill();
        }
    });

    var btn=document.getElementsByClassName("predictBtn")[0];
    var output = document.getElementsByClassName('output');
    var ptxt=document.getElementsByClassName('ptxt')[0];
    var span=document.getElementsByClassName('pspan')[0];
    var num=0;
    var pred=[];
    console.log(ptxt);
    btn.addEventListener('click',()=>{
        if(!o)
        {
            smallCtx.drawImage(largeCanvas, 0, 0, 28, 28);
            const imageData = smallCtx.getImageData(0, 0, 28, 28);
            const data = imageData.data;
            const grayscaleData = [];

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const grayscale = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                grayscaleData.push(grayscale/255.0);
            }

            pred=feedforward(grayscaleData);       
            num= (Math.max(...pred));
            progress.style.width=((num).toPrecision(4)*100).toString()+"%";
            ptxt.innerHTML=(num.toPrecision(2)*100)+"%";
            output[pred.indexOf(num)].style.backgroundColor="#3a5a40";  
            btn.className="closeBtn";
            span.innerHTML="clear";
            o=true;
        }
        else{
            largeCtx.clearRect(0, 0, largeCanvas.width, largeCanvas.height);
            smallCtx.clearRect(0, 0, largeCanvas.width, largeCanvas.height);
            smallCtx.fillStyle = "black";
            smallCtx.fillRect(0, 0, largeCanvas.width, largeCanvas.height);
            btn.className="predictBtn";
            span.innerHTML="predict";
            output[pred.indexOf(num)].style.backgroundColor="#fafafa";
            o=false;
        }

    })
})

