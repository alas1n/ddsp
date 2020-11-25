import React from "react";

const RecorderCmp = (props) => {
  //   console.log(props);
  return (
    <div className="Recorder">
      <button
        onClick={() => props.startRecording()}
        disabled={props.isRecording}
      >
        Record
      </button>
      <button
        onClick={() => props.stopRecording()}
        disabled={!props.isRecording}
      >
        Stop
      </button>
      <audio src={props.blobURL} controls="controls" />
      <button
        onClick={async () => {
          await props.uploadAudio(props.blob);
          console.log("sent");
        }}
      >
        send
      </button>
      {/* <button
        onClick={async () => {
          await props.downloadAudio(props.num);
          console.log("Audio downloaded");
        }}
      >
        getAudo
      </button>
      <audio src={props.blobURL_received[props.num]} controls="controls" /> */}
    </div>
  );
};

export default RecorderCmp;
