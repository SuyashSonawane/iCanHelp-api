$('#speak-btn').click(()=>{
    let text_tbr = `There is no text to be read.`

    if($('#text_tbr').val().length > 0) {
        text_tbr = $('#text_tbr').val();
    }

    let msg = new SpeechSynthesisUtterance(`${text_tbr}`);
    window.speechSynthesis.speak(msg);
})