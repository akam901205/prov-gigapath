import { NextRequest, NextResponse } from 'next/server'

const SIMPATH_API_URL = 'http://localhost:8006' // Use same port as true-tiered API

// Configure route for long-running operations
export const maxDuration = 1800 // 30 minutes - maximum allowed
export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  const requestStartTime = new Date().toISOString()
  try {
    console.log('🚀 Simpath proxy: Starting request processing at', requestStartTime)
    
    let body: any
    let metrics: string[]
    let imageBase64: string
    
    // Try to handle FormData first, fall back to JSON
    const contentType = request.headers.get('content-type') || ''
    
    if (contentType.includes('multipart/form-data')) {
      // Handle FormData like true-tiered analysis
      console.log('📝 Processing FormData request')
      const formData = await request.formData()
      const image = formData.get('image') as File
      metrics = JSON.parse(formData.get('metrics') as string || '["cosine"]')
      
      if (!image) {
        return NextResponse.json({ error: 'No image provided' }, { status: 400 })
      }
      
      console.log('📊 Image file size:', image.size, 'bytes')
      
      // Convert to base64 server-side
      const imageBuffer = await image.arrayBuffer()
      imageBase64 = Buffer.from(imageBuffer).toString('base64')
      
    } else {
      // Handle JSON request (backward compatibility)
      console.log('📝 Processing JSON request')
      const jsonBody = await request.json()
      metrics = jsonBody?.input?.metrics || ['cosine']
      imageBase64 = jsonBody?.input?.image_base64 || ''
      
      if (!imageBase64) {
        return NextResponse.json({ error: 'No image data provided' }, { status: 400 })
      }
      
      console.log('📊 Image base64 length:', imageBase64.length)
    }
    
    console.log('📝 Request processed, metrics:', metrics)
    
    body = {
      input: {
        image_base64: imageBase64,
        metrics: metrics
      }
    }
    
    console.log(`🌐 Forwarding to RunPod API: ${SIMPATH_API_URL}`)
    
    // Allow much longer for large image processing (15 minutes total)
    const startTime = Date.now()
    const quickController = new AbortController()
    const quickTimeoutId = setTimeout(() => quickController.abort(), 1800000) // 30 minutes full timeout
    
    try {
      const response = await fetch(`${SIMPATH_API_URL}/api/simpath-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'NextJS-Proxy/1.0'
        },
        body: JSON.stringify(body),
        signal: quickController.signal
      })
      
      clearTimeout(quickTimeoutId)
      
      const elapsed = Date.now() - startTime
      console.log(`⏱️ RunPod API responded in ${elapsed}ms with status ${response.status}`)
      
      if (response.ok) {
        const result = await response.json()
        return NextResponse.json(result)
      } else {
        // Handle non-OK responses (4xx, 5xx)
        const errorText = await response.text()
        console.error(`❌ RunPod API error ${response.status}: ${errorText}`)
        return NextResponse.json({
          error: 'SimPath analysis failed',
          details: `Status ${response.status}: ${errorText}`
        }, { status: response.status })
      }
    } catch (fetchError) {
      clearTimeout(quickTimeoutId)
      console.error('❌ SimPath analysis failed:', fetchError)
      
      // Return error response for actual failures
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        return NextResponse.json({
          error: 'SimPath analysis timed out',
          message: 'The analysis took longer than 15 minutes. Please try again with a smaller image or fewer similarity metrics.',
          timeout: true
        }, { status: 504 })
      }
      
      return NextResponse.json({
        error: 'SimPath analysis failed',
        details: fetchError instanceof Error ? fetchError.message : 'Unknown error',
        message: 'Failed to process the image. Please try again.'
      }, { status: 500 })
    }
    
    // This code should not be reached due to the try-catch above
    console.log('⚠️ Unexpected code path reached')
    return NextResponse.json({
      error: 'Unexpected processing state'
    }, { status: 500 })

  } catch (error: any) {
    console.error('💥 Simpath proxy error:', error)
    if (error.name === 'AbortError' || error.name === 'TimeoutError' || error.message.includes('timeout')) {
      return NextResponse.json(
        { error: 'SimPath analysis timed out after 30 minutes. This is a complex computation comparing against 2,217 pathology samples across 8 similarity metrics. Please try with a smaller image or fewer metrics.' },
        { status: 504 }
      )
    }
    return NextResponse.json(
      { error: 'Simpath analysis failed', details: error.message },
      { status: 500 }
    )
  }
}