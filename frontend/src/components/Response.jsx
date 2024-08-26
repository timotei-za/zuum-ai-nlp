import { useState } from 'react';
import Message from './Message';
import '../style/response.css';

const Response = ({ display }) => {
    return (
        <div id='response-display'>
                {display.map(
                    (response, index) => {
                        return (
                            <Message response={response} index={index}/>
                        )
                    })
                }
        </div>
    )
}

export default Response;