var role = document.getElementById("agent_call").getAttribute("role")
var debug = document.getElementById("agent_call").getAttribute("debug")
var c = document.getElementById("board");
var ctx = c.getContext("2d");
var debug_text = document.getElementById("debug");
var message = document.getElementById("message");
var game_board_size = 9
var radius = 14;
var blank = 20;
var turn = 1; // 1 black 2 white
var width = (game_board_size - 1) * 32 + blank * 2;
var height = (game_board_size - 1) * 32 + blank * 2;

debug_text.innerHTML = debug;

var boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		boardArray[i][j] = 0.0;
	}
}

var player_pi_colors = [
	"#e0e0e0",
	"#FFF5CC",
	"#FFE670",
	"#FFCC33",
	"#FFAF33",
	"#FF9933",
	"#FF6F33",
	"#FF5500",
	"#E6281E",
	"#C81E14"
];

var enemy_pi_colors = [
	"#e0e0e0",
	"#F7FCF0",
	"#E0F3DB",
	"#CCEBC5",
	"#A8DDB5",
	"#7BCCC4",
	"#4EB3D3",
	"#2B8CBE",
	"#0868AC",
	"#084081"
];

function reqAgent()
{
	var xhr = new XMLHttpRequest();
	
	xhr.onload = function() 
	{
		if(xhr.status == 200) 
		{
			ret = JSON.parse(xhr.responseText);
			message.innerHTML = xhr.responseText;
			
			game_board_size = ret.debug_size

			for (var i = 0; i < game_board_size; i++) 
			{
				for (j = 0; j < game_board_size; j++) 
				{ 
					idx = i + j * game_board_size;
					boardArray[i][j] = ret.debug_values[idx]
				}
			}
			
			message.innerHTML = role + ' : ' + ret.message;

			updateBoard();	
		}
	};

	xhr.open('GET', 'http://127.0.0.1:5000/agent?role=' + role + '&debug=' + debug, true);
	xhr.send();
}

function clearBoard()
{
	for (var i = 0; i < game_board_size; i++) 
	{
		for (j = 0; j < game_board_size; j++) 
		{ 
			boardArray[i][j] = 0.0;
		}
	}
}

function updateBoard()
{
	// board fill color
	ctx.fillStyle="#efefef";
	ctx.fillRect(0, 0, width, height);

	// board draw line
	ctx.strokeStyle = '#ffffff';
	ctx.lineWidth = 1

	for (i = 0; i < game_board_size; i++) 
	{ 
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
	
	ctx.fillStyle="#000000";
	ctx.font="11px Arial";

	max_pi_val = 0.0;
	min_pi_val = 1.0;

	for (i = 0; i < game_board_size; i++) 
	{ 
		for (j = 0; j < game_board_size; j++)
		{
			pi_val = boardArray[i][j];

			if (pi_val > max_pi_val)
			{
				max_pi_val = pi_val
			}

			if (pi_val < min_pi_val && pi_val != 0.0)
			{
				min_pi_val = pi_val
			}
		}
	}
	
	if (max_pi_val == 0)
	{
		max_pi_val = 1.0;
	}

	// board draw clicked
	for (i = 0; i < game_board_size; i++) 
	{ 
		for (j = 0; j < game_board_size; j++)
		{
			pi_val = boardArray[i][j];

			if (pi_val == 0.0)
			{
				pi_val_idx = 0;
			}
			else
			{
				pi_val_idx = parseInt((pi_val - min_pi_val) * 8.0 / (max_pi_val - min_pi_val));
				pi_val_idx = pi_val_idx + 1;
			}

			if(role == "player")
			{
				ctx.fillStyle = player_pi_colors[pi_val_idx];
			}
			else
			{
				ctx.fillStyle = enemy_pi_colors[pi_val_idx];
			}

			ctx.beginPath();
			ctx.arc(blank + i * 32, blank + j * 32, radius, 0, 2*Math.PI);
			ctx.fill();
			ctx.stroke();
			
			ctx.fillStyle="#000000";

			if (debug == "pi")
			{
				pi_val_cut = pi_val.toExponential(1);
				pi_val_str = pi_val_cut.toString();					
				pi_val_str_lines = pi_val_str.split("e")
				ctx.textAlign="center"; 
				ctx.fillText(pi_val_str_lines[0], blank + i * 32, blank - 1 + j * 32);
				ctx.fillText("e" + pi_val_str_lines[1], blank + i * 32, blank + 9 + j * 32);
			}
			else
			{
				pi_val_str = pi_val.toString();	
				ctx.textAlign="center"; 
				ctx.fillText(pi_val_str, blank + i * 32, blank + 4 + j * 32);
			}
		}
	}
}

updateBoard();
setInterval(reqAgent, 500);//1000 is miliseconds