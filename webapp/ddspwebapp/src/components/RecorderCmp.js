import React from "react";

const RecorderCmp = (props) => {
  //   console.log(props);
  return (
    <div className="Recorder">
      <button
        onClick={() => props.stertRecording(props.num)}
        disabled={props.isRecording}
      >
        Record
      </button>
      <button
        onClick={() => props.stopRecording(props.num)}
        disabled={!props.isRecording}
      >
        Stop
      </button>
      <audio src={props.blobURL} controls="controls" />
    </div>
  );
};

export default RecorderCmp;
