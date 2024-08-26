import { useState, useEffect } from 'react';
import { ChakraProvider, Heading, Stack, Box, Flex, Image, Link } from '@chakra-ui/react';
import Chatbox from './components/Chatbox';
import Response from './components/Response';
import wavetmsLogo from './assets/wavetmslogo.png'
import './App.css'

function App() {
  const [display, setDisplay] = useState([])

  useEffect(() => {
    console.log(display)
  }, [display])

  return (
    <ChakraProvider> 
      <Box minH="100vh">    

        <Stack align="center">
        <Link href="https://www.wavetms.com/" mt="5%" width="250px" height="auto" mb="5%" isExternal>
            <Image src={wavetmsLogo} />
          </Link>
          <Response display={display} />
          <Box mt="1%" w="80%" mb="5rem">
            <Chatbox display={display} setDisplay={setDisplay} />
          </Box>
        </Stack> 
      </Box> 
    </ChakraProvider>
  )
}

export default App;