var role = document.getElementById("agent_call").getAttribute("role")
var c = document.getElementById("board");
var ctx = c.getContext("2d");
var message = document.getElementById("message");
var pi_size = 9
var radius = 14;
var blank = 20;
var turn = 1; // 1 black 2 white
var width = (game_board_size - 1) * 32 + blank * 2;
var height = (game_board_size - 1) * 32 + blank * 2;

var boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		boardArray[i][j] = 0.0;
	}
}

function reqAgent()
{
	var xhr = new XMLHttpRequest();
	
	xhr.onload = function() 
	{
		if(xhr.status == 200) 
		{
			ret = JSON.parse(xhr.responseText);
			
			pi_size = ret.pi_size

			for (var i = 0; i < pi_size; i++) 
			{
				for (j = 0; j < pi_size; j++) 
				{ 
					idx = i + j * pi_size;
					boardArray[i][j] = ret.pi_values[idx]
				}
			}

			message.innerHTML = ret.message;

			updateBoard();	
		}
	};

	xhr.open('GET', 'http://127.0.0.1:5000/agent?role=' + role, true);
	xhr.send();
}

function clearBoard()
{
	for (var i = 0; i < pi_size; i++) 
	{
		for (j = 0; j < pi_size; j++) 
		{ 
			boardArray[i][j] = 0;
		}
	}

}

function updateBoard(){
	// board fill color
	ctx.fillStyle="#ffcc66";
	ctx.fillRect(0, 0, width, height);

	// board draw line
	// ctx.strokeStyle="#333300";
	// ctx.fillStyle="#333300";
	ctx.strokeStyle = 'black';
	ctx.fillStyle="#FF0000";
	ctx.lineWidth = 1
	for (i = 0; i < game_board_size; i++) { 
		// horizontal line draw
		ctx.beginPath();
		ctx.moveTo(blank + i * 32 + 0.5, blank);
		ctx.lineTo(blank + i * 32 + 0.5, width - blank + 0.5);
		ctx.stroke();

		// vertical line draw
		ctx.beginPath();
		ctx.moveTo(blank, blank + i * 32 + 0.5);
		ctx.lineTo(height - blank, blank + i * 32 + 0.5);
		ctx.stroke();
	}

	/*
	// board draw point
	var circleRadius = 3;
	for (i = 0; i < 3; i++) { 
		for (j = 0; j < 3; j++) { 
			// board circle draw
			ctx.beginPath();
			ctx.arc(blank + 3 * 32 + i * 6 * 32, blank + 3 * 32  + j * 6 * 32, circleRadius, 0, 2*Math.PI);
			ctx.fill();
			ctx.stroke();
		}
	}
	*/

	// board draw clicked
	for (i = 0; i < game_board_size; i++) { 
		for (j = 0; j < game_board_size; j++) {
			if (boardArray[i][j] == 1) {
				ctx.beginPath();
				ctx.strokeStyle="#000000";
				ctx.fillStyle="#000000";
				ctx.arc(blank + i * 32, blank + j * 32, radius, 0, 2*Math.PI);
				ctx.fill();
				ctx.stroke();
			} else if (boardArray[i][j] == 2){
				ctx.beginPath();
				ctx.strokeStyle="#ffffff";
				ctx.fillStyle="#ffffff";
				ctx.arc(blank + i * 32, blank + j * 32, radius, 0, 2*Math.PI);
				ctx.fill();
				ctx.stroke();
			}
		}
	}


}

updateBoard();
setInterval(reqAgent, 500);//1000 is miliseconds

/* Mouse Event */
function getMousePos(canvas, evt) {
	var rect = canvas.getBoundingClientRect();
	return {
	  x: evt.clientX - rect.left,
	  y: evt.clientY - rect.top
	};
}

function getMouseRoundPos(xPos, yPos){
	var x = (xPos - blank) / 32;
	var resultX = Math.round(x);
	var y = (yPos - blank) / 32;
	var resultY = Math.round(y);

	return {
		x: resultX,
		y: resultY
	};
}

c.addEventListener('mousemove', function(evt) {
	var mousePos = getMousePos(c, evt);
	drawNotClicked(mousePos.x, mousePos.y);
}, false);

c.addEventListener('mousedown', function(evt) {
	var mousePos = getMousePos(c, evt);
	isClicked(mousePos.x, mousePos.y);
}, false);

function drawNotClicked(xPos, yPos){
	resultPos = getMouseRoundPos(xPos, yPos);

	if (resultPos.x > -1 && resultPos.x < game_board_size && resultPos.y > -1
	 && resultPos.y < game_board_size && boardArray[resultPos.x][resultPos.y] == 0){
		updateBoard();
		ctx.beginPath();
		ctx.globalAlpha=0.8;
		if (turn < 2) {
			ctx.strokeStyle="#000000";
			ctx.fillStyle="#000000";
		} else {
			ctx.strokeStyle="#ffffff";
			ctx.fillStyle="#ffffff";	
		}
		ctx.arc(blank + resultPos.x * 32, blank + resultPos.y * 32, radius, 0, 2*Math.PI);
		ctx.fill();
		ctx.stroke();
		ctx.globalAlpha=1;
	}
};

function isClicked(xPos, yPos){
	resultPos = getMouseRoundPos(xPos, yPos);
	if (resultPos.x > -1 && resultPos.x < game_board_size && resultPos.y > -1
	 && resultPos.y < game_board_size && boardArray[resultPos.x][resultPos.y] == 0){
		// boardArray[resultPos.x][resultPos.y] = turn;
		// checkOmok(turn, resultPos.x, resultPos.y);
		// turn = 3 - turn; //turn change
	}

	action_idx = resultPos.x + resultPos.y * game_board_size;

	reqAction(action_idx)

	updateBoard();
}

/* is Omok?? */
function checkOmok(turn, xPos, yPos){
	if (addOmok(turn, xPos, yPos, -1, -1) + addOmok(turn, xPos, yPos, 1, 1) == 4) alert("end");
	if (addOmok(turn, xPos, yPos, 0, -1) + addOmok(turn, xPos, yPos, 0, 1) == 4) alert("end");
	if (addOmok(turn, xPos, yPos, 1, -1) + addOmok(turn, xPos, yPos, -1, 1) == 4) alert("end");
	if (addOmok(turn, xPos, yPos, -1, 0) + addOmok(turn, xPos, yPos, 1, 0) == 4) alert("end");
}

function addOmok(turn, xPos, yPos, xDir, yDir){
	if (xPos + xDir < 0) return 0;
	if (xPos + xDir > game_board_size - 1) return 0;
	if (yPos + yDir < 0) return 0;
	if (yPos + yDir > game_board_size - 1) return 0;

	if (boardArray[xPos + xDir][yPos + yDir] == turn) {
		return 1 + addOmok(turn, xPos + xDir, yPos + yDir, xDir, yDir);
	} else {
		return 0;
	}
}
