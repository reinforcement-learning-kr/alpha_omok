var ctx = document.getElementById("board").getContext("2d");
var ctx_pap = document.getElementById("player_agent_p").getContext("2d");
var ctx_pav = document.getElementById("player_agent_visit").getContext("2d");
var ctx_eap = document.getElementById("enemy_agent_p").getContext("2d");
var ctx_eav = document.getElementById("enemy_agent_visit").getContext("2d");

var game_board_size = 9
var radius = 14;
var blank = 20;
var turn = 1; // 1 black 2 white
var width = (game_board_size - 1) * 32 + blank * 2;
var height = (game_board_size - 1) * 32 + blank * 2;

var board_message = document.getElementById("board_message");
var player_message = document.getElementById("player_message");
var enemy_message = document.getElementById("enemy_message");

var boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		boardArray[i][j] = 0;
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


var player_p_boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    player_p_boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		player_p_boardArray[i][j] = 0;
	}
}

var player_visit_boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    player_visit_boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		player_visit_boardArray[i][j] = 0;
	}
}

var enemy_p_boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    enemy_p_boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		enemy_p_boardArray[i][j] = 0;
	}
}

var enemy_visit_boardArray = new Array(game_board_size); 
for (var i = 0; i < game_board_size; i++) {
    enemy_visit_boardArray[i] = new Array(game_board_size);
    for (j = 0; j < game_board_size; j++) { 
		enemy_visit_boardArray[i][j] = 0;
	}
}
var player_move = [];
var player_value = [];
var enemy_move = [];
var enemy_value = [];

function updateBoard(ret)
{
    if (ret.curr_turn == 0) // black turn: 0, white turn: 1
    {
        turn = 1 // black
    }
    else 
    {
        turn = 2 // white
    }

    game_board_size = ret.game_board_size

    for (var i = 0; i < game_board_size; i++) 
    {
        for (j = 0; j < game_board_size; j++) 
        { 
            idx = i + j * game_board_size;

            switch (ret.game_board_values[idx])
            {
                case 1:
                    boardArray[i][j] = 1;
                    break;
                case -1:
                    boardArray[i][j] = 2;
                    break;		
                case 0:
                    boardArray[i][j] = 0;
                    break;		
            }
        }
    }
}

function updateStatusBoard(agent, item_name)
{
    if (agent == "player" && item_name == "p")
    {
        target_boardArrary = player_p_boardArray;
        source_values = ret.player_agent_p_values;
    }
    else if(agent == "player" && item_name == "visit")
    {
        target_boardArrary = player_visit_boardArray;
        source_values = ret.player_agent_visit_values;
    }
    else if(agent == "enemy" && item_name == "p")
    {
        target_boardArrary = enemy_p_boardArray;
        source_values = ret.enemy_agent_p_values;
    }
    else if(agent == "enemy" && item_name == "visit")
    {
        target_boardArrary = enemy_visit_boardArray;
        source_values = ret.enemy_agent_visit_values;
    }  
    
    for (var i = 0; i < game_board_size; i++) 
    {
        for (j = 0; j < game_board_size; j++) 
        { 
            idx = i + j * game_board_size;
            target_boardArrary[i][j] = source_values[idx];
        }
    }
}

function updateVPlot(ret)
{
    player_move = ret.player_agent_moves;
    player_value = ret.player_agent_values;
    enemy_move = ret.enemy_agent_moves;
    enemy_value = ret.enemy_agent_values;
}

function renderBoard(){
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
    
    board_message.innerHTML = "AlphaZero vs AlphaZero";
}

function renderStatusBoard(agent, item_name)
{
    if (agent == "player" && item_name == "p")
    {
        ctx_target = ctx_pap;
    }
    else if(agent == "player" && item_name == "visit")
    {
        ctx_target = ctx_pav;
    }
    else if(agent == "enemy" && item_name == "p")
    {
        ctx_target = ctx_eap;
    }
    else if(agent == "enemy" && item_name == "visit")
    {
        ctx_target = ctx_eav;
    }  

	// board fill color
	ctx_target.fillStyle="#efefef";
	ctx_target.fillRect(0, 0, width, height);

	// board draw line
	ctx_target.strokeStyle = '#ffffff';
	ctx_target.lineWidth = 1

    if (agent == "player" && item_name == "p")
    {
        target_boardArrary = player_p_boardArray;
    }
    else if(agent == "player" && item_name == "visit")
    {
        target_boardArrary = player_visit_boardArray;
    }
    else if(agent == "enemy" && item_name == "p")
    {
        target_boardArrary = enemy_p_boardArray;
    }
    else if(agent == "enemy" && item_name == "visit")
    {
        target_boardArrary = enemy_visit_boardArray;
    }        

	for (i = 0; i < game_board_size; i++) 
	{ 
		// horizontal line draw
		ctx_target.beginPath();
		ctx_target.moveTo(blank + i * 32 + 0.5, blank);
		ctx_target.lineTo(blank + i * 32 + 0.5, width - blank + 0.5);
		ctx_target.stroke();

		// vertical line draw
		ctx_target.beginPath();
		ctx_target.moveTo(blank, blank + i * 32 + 0.5);
		ctx_target.lineTo(height - blank, blank + i * 32 + 0.5);
		ctx_target.stroke();
	}
	
	ctx_target.fillStyle="#000000";
	ctx_target.font="11px Arial";

	max_pi_val = 0.0;
	min_pi_val = 1.0;

	for (i = 0; i < game_board_size; i++) 
	{ 
		for (j = 0; j < game_board_size; j++)
		{
			pi_val = target_boardArrary[i][j];

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
			pi_val = target_boardArrary[i][j];

			if (pi_val == 0.0)
			{
				pi_val_idx = 0;
			}
			else
			{
				pi_val_idx = parseInt((pi_val - min_pi_val) * 8.0 / (max_pi_val - min_pi_val));
				pi_val_idx = pi_val_idx + 1;
			}

			if(agent == "player")
			{
				ctx_target.fillStyle = player_pi_colors[pi_val_idx];
			}
			else
			{
				ctx_target.fillStyle = enemy_pi_colors[pi_val_idx];
			}

			ctx_target.beginPath();
			ctx_target.arc(blank + i * 32, blank + j * 32, radius, 0, 2*Math.PI);
			ctx_target.fill();
			ctx_target.stroke();
			
			ctx.fillStyle="#000000";

			if (item_name == "pi")
			{
				pi_val_cut = pi_val.toExponential(1);
				pi_val_str = pi_val_cut.toString();					
				pi_val_str_lines = pi_val_str.split("e")
				ctx_target.textAlign="center"; 
				ctx_target.fillText(pi_val_str_lines[0], blank + i * 32, blank - 1 + j * 32);
				ctx_target.fillText("e" + pi_val_str_lines[1], blank + i * 32, blank + 9 + j * 32);
			}
			else
			{
				pi_val_str = pi_val.toString();	
				ctx_target.textAlign="center"; 
				ctx_target.fillText(pi_val_str, blank + i * 32, blank + 4 + j * 32);
			}
		}
	}
}

function renderVPlot()
{
    var trace1 = {
        x: player_move, 
        y: player_value, 
        type: 'scatter',
        fill: 'tozeroy',
        name: 'player'
    };
    
    var trace2 = {
        x: enemy_move, 
        y: enemy_value, 
        type: 'scatter',
        fill: 'tozeroy',
        name: 'enemy'
    };

    var data = [trace1, trace2];
    Plotly.newPlot('v_monitoring', data);
}

function reqPeriodicStatus()
{
    var xhr = new XMLHttpRequest();
    
    xhr.onload = function() 
    {
        if(xhr.status == 200) 
        {
            ret = JSON.parse(xhr.responseText);
            
            updateBoard(ret);
            updateStatusBoard("player", "p");
            updateStatusBoard("player", "visit");
            updateStatusBoard("enemy", "p");
            updateStatusBoard("enemy", "visit");

            updateVPlot(ret);

            renderBoard();
            renderVPlot();
            renderStatusBoard("player", "p");
            renderStatusBoard("player", "visit");
            renderStatusBoard("enemy", "p");
            renderStatusBoard("enemy", "visit");

        }
    };

    xhr.open('GET', 'http://127.0.0.1:5000/periodic_status', true);
    xhr.send();
}

function reqPromptStatus()
{
    var xhr = new XMLHttpRequest();
    
    xhr.onload = function() 
    {
        if(xhr.status == 200) 
        {
            ret = JSON.parse(xhr.responseText);

            player_message.innerHTML = ret.player_message;
            enemy_message.innerHTML = ret.enemy_message;	
        }
    };

    xhr.open('GET', 'http://127.0.0.1:5000/prompt_status', true);
    xhr.send();
}

function clearBoard()
{
	for (var i = 0; i < game_board_size; i++) 
	{
		for (j = 0; j < game_board_size; j++) 
		{ 
			boardArray[i][j] = 0;
		}
	}

}

renderBoard();
renderVPlot();

setInterval(reqPeriodicStatus, 500);
setInterval(reqPromptStatus, 1);
