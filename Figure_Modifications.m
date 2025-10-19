% MATLAB Script to Format ALL Open Figures for Scientific Publication
%
% This script applies publication-ready formatting to all line, scatter, 
% and axis elements in ALL figures currently open in the MATLAB environment.
%
% Formatting includes:
% - Setting the font to Times New Roman.
% - Increasing font sizes for labels and ticks (12pt/14pt).
% - Increasing the line width of the plotted data (2.5pt).
% - Cleaning the figure background and axes box.

% --- 0. FUNCTION DEFINITION ---
function format_figure_for_publication(h_fig)
% Helper function to apply styling to a single figure handle h_fig

    % Ensure figure background is white
    set(h_fig, 'Color', 'w');

    % Find all axes (including subplots) within the figure, excluding legends
    h_axes = findobj(h_fig, 'Type', 'axes', '-not', 'Tag', 'legend');
    
    for h_ax = h_axes'
        % --- A. AXES AND FONT SETTINGS ---

        % Set the default font and size for axes elements (ticks, labels)
        set(h_ax, ...
            'FontName', 'Times New Roman', ...
            'FontSize', 12, ...             
            'Box', 'off', ...                % Remove the top and right borders/box
            'TickDir', 'out', ...            % Place ticks outside
            'Color', 'w', ...                % Ensure axes background is white
            'LineWidth', 1.5);               % Thicker axis lines (the frame/box lines)

        % Increase the font size for the Title, XLabel, and YLabel specifically (14pt)
        set(get(h_ax, 'Title'), 'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
        set(get(h_ax, 'XLabel'), 'FontSize', 14, 'FontName', 'Times New Roman');
        set(get(h_ax, 'YLabel'), 'FontSize', 14, 'FontName', 'Times New Roman');
        
        % Remove grid lines for a cleaner aesthetic
        grid(h_ax, 'off');

        % --- B. LINE THICKNESS ---

        % Find all plotted line, scatter, and errorbar objects in the current axes
        h_plots = findobj(h_ax, 'Type', 'line', '-or', 'Type', 'scatter', '-or', 'Type', 'errorbar'); 
        line_width_value = 2.5;

        for h_plot = h_plots'
            % Filter out internal MATLAB graphics elements (like axis lines or colorbars)
            if ~strcmp(get(h_plot, 'Tag'), 'TMW_colorBar') && ~strcmp(get(h_plot, 'Tag'), 'TMW_Axis')
                % Apply the increased line width
                set(h_plot, 'LineWidth', line_width_value);
            end
        end

        % --- C. LEGEND FONT ---
        % Check for a legend on the current axes
        h_legend = legend(h_ax);
        if ~isempty(h_legend) && isvalid(h_legend)
            set(h_legend, 'FontSize', 12, 'FontName', 'Times New Roman');
        end
    end
end
% --- END OF FUNCTION DEFINITION ---

% --- 1. SCRIPT EXECUTION: APPLY TO ALL OPEN FIGURES ---

% Find all currently open figure handles
h_figures = get(0, 'Children');

if isempty(h_figures)
    warning('No figures are currently open. Please open a figure (e.g., using plot() or surf()) and re-run this script.');
else
    % Iterate over all open figures and apply the formatting function
    for h_fig = h_figures'
        format_figure_for_publication(h_fig);
    end
    disp(['Publication-ready formatting applied to ', num2str(length(h_figures)), ' open figure(s).']);
end

% --- 2. EXPORTING THE FIGURE (Best Practice) ---
% To ensure high quality, figures should be exported in a vector format (.eps, .pdf)
% or a high-resolution raster format (.tiff, .png).

% Example command to save the figure as a high-quality PDF:
% print('formatted_figure', '-dpdf', '-r300'); % -r300 sets 300 DPI resolution
%
% Example command to save the figure as a high-quality TIFF:
% print('formatted_figure', '-dtiff', '-r300');
