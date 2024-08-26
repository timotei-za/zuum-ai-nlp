import { useState } from 'react';
import axios from 'axios';
import '../style/chatbox.css';
import { Spinner, Flex, Box } from '@chakra-ui/react'

const Chatbox = ({ display, setDisplay }) => {
    const [userQuery, setUserQuery] = useState(''); // rewrite using form lib later
    const [isLoading, setIsLoading] = useState(false);

    const getResponse = async (query) => {
        const res = await axios.get(`/api/query?query=${query}`);
        setIsLoading(false);
        return res['data']
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);

        let query = userQuery;
        setDisplay([...display, query]);
        setUserQuery('');
        
        let chatbotResponse = await getResponse(query);
        console.log(chatbotResponse);
        setDisplay(display => [...display, chatbotResponse]); 
    }

    return (
        <>
        <Flex w="100%">
            <Box w="100%">
                <Box w="100%">
                    <form id="form" onSubmit={handleSubmit}>
                        <input
                            placeholder="Start chatting here" 
                            id="chatbox-input"
                            onChange={event => setUserQuery(event.target.value)} 
                            value={userQuery}
                        />
                    </form>
                </Box>

                
            </Box>
            </Flex>
        </>
    );
}

export default Chatbox;
