var timer = 0
const stopTimer = _ => clearInterval(timer);

var frameBufferTimeInMs = 2 * 1000;
var frameArray = [];

const pause = () => {
    stopTimer();
    for (let i = 0; i < frameArray.length; i++) {
        clearTimeout(frameArray[i].timeout);
    }
    showReport();
}
const showReport = () => {

    //ajax call to the the 3rd and 4th api
    var REST_CALL = "http://127.0.0.1:8887/get-report";

    // var REST_CALL = "https://www.boredapi.com/api/activity"
    const fetchFinalData = _ => {
        $.ajax(
            {
                type: 'GET',
                url: REST_CALL,
                success: function (data) {
                    document.getElementById("type-data").innerHTML = "Duration";
                    if ($("#table-content tbody tr").length > 0) {
                        console.log("Removing rows")
                        $("#table-content tr:not(:first)").remove();
                    }
                    for (var key of Object.keys(data.report)) {
                        var col1 = key;
                        var col2 = data.report[col1]
                        $('#table-content').append(`<tr>
                       <td> ${col1}</td>
                       <td>${col2} seconds</td
                       </tr>`);
                    }

                },
                error: function (textStatus, errorThrown) {
                    alert("Error!!")
                }
            }
        );
    }

    fetchFinalData()

    const DOMStrings = {
        download_button: document.getElementById("download-report"),
        report_heading: document.getElementById("report_heading")
    };

    DOMStrings.download_button.disabled = false;
    DOMStrings.report_heading.innerHTML = "Report"
}

const getData = () => {
    //ajax call to the api for frames 2nd api
    //update frame and table
    // var SERVER_URL = window.location.protocol + "//" + window.location.host;
    document.getElementById("loader").hidden = true;
    document.getElementById("frame-img").hidden = false;

    var REST_CALL = "http://127.0.0.1:8887/get-current-activity";

    const fetchApiData = _ => {
        $.ajax(
            {
                type: 'GET',
                url: REST_CALL,
                success: function (data) {
                    if (data === undefined || data.length == 0) {
                        // $("#frame-img").attr('src', 'blackBG.jpg');
                        stopTimer()
                    }

                    // Render N frames in 2 second
                    let calcTimeInMs = Math.floor(frameBufferTimeInMs / data.length);
                    frameArray = [];
                    for (let i = 0; i < data.length; i++) {
                        frameArray.push({ timeout: setTimeout(_ => setImgData(data[i]), (i + 1) * calcTimeInMs) });
                    }
                },
                error: function (textStatus, errorThrown) {
                    alert("Error!!")
                    $("#frame-img").attr('src', 'blackBG.jpg');
                    stopTimer()
                }
            }
        );
    }

    const setImgData = (apidata) => {

        $("#frame-img").attr('src', apidata.current_frame);

        if ($("#table-content tbody tr").length > 0) {
            $("#table-content tr:not(:first)").remove();
        }

        for (var key of Object.keys(apidata.current_activity)) {
            var col1 = key;
            var col2 = apidata.current_activity[col1]
            $('#table-content').append(`<tr>
          <td> ${col1}</td>
           <td>${col2}</td
           </tr>`);
        }
    }



    const api_poll = _ => setInterval(_ => fetchApiData(), frameBufferTimeInMs);
    timer = api_poll();
    const DOMStrings = {
        download_button: document.getElementById("download-report"),
        report_heading: document.getElementById("report_heading")
    };

    DOMStrings.download_button.disabled = true;
    DOMStrings.report_heading.innerHTML = "Current Activity"
}

const downloadData = () => {
    // for csv download

    // var csv='';
    // Object.keys(finaldata).forEach(function(key) {
    //       csv+=key+","+finaldata[key]+","
    //       csv += "\n";
    //   })
    // let file = new File([csv], "fe.csv", {
    //     type: "text/plain",
    // });

    // filename="namet"
    // saveAs(file, filename + ".csv");

    var sTable = document.getElementById('table-content-div').innerHTML;

    var style = "<style>";
    style = style + "table {width: 100%;font: 17px Calibri;}";
    style = style + "table, th, td {border: solid 1px #DDD; border-collapse: collapse;";
    style = style + "padding: 2px 3px;text-align: center;}";
    style = style + "</style>";

    // CREATE A WINDOW OBJECT.
    var win = window.open('', '', 'height=700,width=700');

    win.document.write('<html><head>');
    win.document.write('<title>Profile</title>');   // <title> FOR PDF HEADER.
    win.document.write(style);          // ADD STYLE INSIDE THE HEAD TAG.
    win.document.write('</head>');
    win.document.write('<body>');
    win.document.write(sTable);         // THE TABLE CONTENTS INSIDE THE BODY TAG.
    win.document.write('</body></html>');

    win.document.close(); 	// CLOSE THE CURRENT WINDOW.

    win.print();    // PRINT THE CONTENTS.
    win.close();

}



$('form').submit(function (e) {
    e.preventDefault();

    var REST_CALL = "http://127.0.0.1:8887/upload-video"

    var fd = new FormData();
    fd.append('file', $('#file')[0].files[0]);
    $.ajax({
        url: REST_CALL,
        data: fd,
        processData: false,
        contentType: false,
        type: 'POST',
        success: function (data) {
            alert(data.message);
            $('form').find('input:file').val('');
            document.getElementById("loader").hidden = false;
            document.getElementById("frame-img").hidden = true;
        },
        error: function (textStatus, errorThrown) {
            alert("Error!!")
            $('form').find('input:file').val('');
        }
    });
});