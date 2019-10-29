library(shiny)
library(shinydashboard)
library(png)
library(grid)

id_cbm <- feather::read_feather('06-img/id_cbm.feather')
id_opt <- feather::read_feather('06-img/id_optimizer.feather')
vec_cbm <- id_cbm$cbm_idx
names(vec_cbm) <- id_cbm$cbm
vec_opt <- id_opt$opt_idx
names(vec_opt) <- id_opt$optimizer

ui <- dashboardPage(
    dashboardHeader(title='Experiment 04'),
    dashboardSidebar(
        selectInput('cbm', 'Model Parameters', choices = vec_cbm),
        selectInput('opt', 'Optimizer', choices = vec_opt)
    ),
    dashboardBody(
        plotOutput('loss'),
        plotOutput('angle')
    )
)

server <- function(input, output) {
    output$loss <- renderPlot({
        grid.raster(readPNG(glue::glue('06-img/{as.integer(input$opt)}_{as.integer(input$cbm)}_loss.png')))
    })
    output$angle <- renderPlot({
        grid.raster(readPNG(glue::glue('06-img/{as.integer(input$opt)}_{as.integer(input$cbm)}_angles.png')))
    })
}

shinyApp(ui, server)