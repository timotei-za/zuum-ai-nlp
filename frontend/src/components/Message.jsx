import '../style/message.css';

const Message = ({ response, index }) => {
    if (Array.isArray(response)) {
        let returnResponse = 
        <div id="table-container">
            <table id="table">
                <thead>
                    <tr>
                        <th>Series Description</th>
                        <th>Value</th>
                        <th>Surcharge</th>
                    </tr>
                </thead>
                <tbody>
                    {response.map((element, id)=> {
                        return (<tr>
                            <td>{element['series-description']}</td>
                            <td>{element['value']}</td>
                            <td>{element['surcharge']}</td>
                        </tr>);
        
                    })}
                </tbody>
            </table>
        </div>
        return returnResponse;
        
    } else {
        let returnResponse;
        if (index % 2 === 0) {
            returnResponse = (<p style={{ textAlign: 'right'}}>
                                {response}
                            </p>)
        }
        else if (index % 2 === 1) {
            returnResponse = (
                <div id="myfavoritediv">
                    <img id="zuumicon" src={'./src/assets/zuumicon.png'} />
                    <p style={{ textAlign: 'left'}}>
                        {response}
                    </p>
                </div>)
        }

        return (
        <div id="message-box" style={{display: 'flex', justifyContent:  index % 2 === 0 ? 'flex-end' : 'flex-start' }} key={index}>
            { response &&
                returnResponse
            }
        </div>
        )
    }
}

export default Message;